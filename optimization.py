import time
from func import *
from model.model_loader import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import copy
import scipy.io as sio


def ADMM_Iter(meas, Phi, truth_tensor, args):
    # -------------- Initialization --------------#
    x0 = shift_back(At(meas, Phi), args.step)
    z, u = x0.to(device), torch.zeros_like(x0).to(device)
    Phi_sum = torch.sum(Phi, 2)
    Phi_sum[Phi_sum == 0] = 1
    im_input = get_input([1, args.ip_BI, x0.shape[0], x0.shape[1]]).to(device)
    best_PSNR, iter_num, lambda_ = 0, args.iter_num, args.lambda_
    net_input_kernel = get_noise(5, 'noise', (1, 1)).type(torch.cuda.FloatTensor)
    net_input_kernel.squeeze_()

    # ---------------- Iteration ----------------#
    for iter in range(iter_num):
        x = z.to(device) - u.to(device)
        x = x + shift_back(At((meas - A(shift(x, args.step), Phi)) / (Phi_sum + lambda_), Phi), args.step)
        z = x + u
        if args.method == 'SK-LCTC':
            z = SK_LCTC(meas, Phi, z.permute(2, 0, 1).unsqueeze(0), truth_tensor, im_input, net_input_kernel, args)
        elif args.method == 'LCTC':
            z = LCTC(meas, Phi, z.permute(2, 0, 1).unsqueeze(0), truth_tensor, im_input, args)
        else:
            z = MK_LCTC(meas, Phi, z.permute(2, 0, 1).unsqueeze(0), truth_tensor, im_input, net_input_kernel, args)
        u = u + (x.to(device) - z.to(device))

        # --------------- Evaluation ---------------#
        psnr_x = calculate_psnr_tensor(truth_tensor, z.permute(2, 0, 1).squeeze(0))
        print('Iter {} | PSNR = {:.2f}dB'.format(iter, psnr_x))
    return z


def LCTC(meas, Phi, z, truth_tensor, im_input, args):
    torch.backends.cudnn.benchmark = True
    iter_num = args.LR_iter
    _, B, _, _ = truth_tensor.shape
    best_loss = float('inf')
    loss_l1 = torch.nn.L1Loss().to(device)
    im_net = CTC_model_load(args.ip_BI, B)

    save_model_weight = False if args.iter_num == 1 else True
    if os.path.exists('Results/model_weights.pth'):
        im_net[0].load_state_dict(torch.load('Results/model_weights.pth'))
        print('----------------------- Load model weights -----------------------')
        iter_num, save_model_weight = args.R_iter, True

    im_net[0].train()
    input_params = [im_input]
    im_input_temp = im_input.clone()
    net_params = list(im_net[0].parameters())
    params = net_params + input_params
    optimizer = torch.optim.Adam([{'params': params, 'lr': 1e-3}])

    for idx in range(iter_num):
        im_input_perturbed = im_input + im_input_temp.normal_() * 0.033
        model_out = im_net[0](im_input_perturbed)  # DIP
        pred_meas = A(shift(model_out.squeeze(0).permute(1, 2, 0), 2).to(device), Phi.to(device))
        if z == None:
            loss = args.lambda_R * loss_l1(meas, pred_meas)
        else:
            loss = args.lambda_R * loss_l1(meas, pred_meas)

        loss_tv = loss_l1(im_input, torch.zeros_like(im_input))
        loss += args.lambda_ * loss_tv
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_hs_recon = model_out.detach()
            if save_model_weight == True:
                torch.save(im_net[0].state_dict(), 'Results/model_weights.pth')

        if (idx + 1) % 100 == 0:
            video_gen(model_out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
            PSNR = calculate_psnr_tensor(truth_tensor, model_out.squeeze(0))
            print(
                'Iter {}, x_loss:{:.3f}, s_loss:{:.3f}, PSNR:{:.2f}'.format(idx + 1, loss.item(), loss_tv.item(), PSNR))

    return best_hs_recon.squeeze(0).permute(1, 2, 0)


def SK_LCTC(meas, Phi, z, truth_tensor, im_input, net_input_kernel, args):
    torch.backends.cudnn.benchmark = True
    iter_num = args.LR_iter
    _, B, _, _ = truth_tensor.shape
    best_loss = float('inf')
    loss_l1 = torch.nn.L1Loss().to(device)

    im_net = CTC_model_load(args.ip_BI, B)

    net_kernel = fcn(5, 5)
    net_kernel = net_kernel.type(torch.cuda.FloatTensor)

    save_model_weight = False if args.iter_num == 1 else True
    if os.path.exists('Results/model_weights.pth'):
        im_net[0].load_state_dict(torch.load('Results/model_weights.pth'))
        print('----------------------- Load model weights -----------------------')
        iter_num, save_model_weight = args.R_iter, True

    if os.path.exists('Results/kernel_weights.pth'):
        net_kernel.load_state_dict(torch.load('Results/kernel_weights.pth'))
        print('----------------------- Load kernel weights -----------------------')

    im_net[0].train()
    net_kernel.train()

    input_params = [im_input]
    im_input_temp = im_input.clone()
    net_params = list(im_net[0].parameters())
    kernel_params = list(net_kernel.parameters())
    optimizer = torch.optim.Adam([
        {'params': net_params, 'lr': 1e-3},  # DIP 网络（im_net）更新正常
        {'params': input_params, 'lr': 1e-3},  # 输入噪声部分
        {'params': kernel_params, 'lr': 1e-3},
    ])

    # flops, model_size = profile(im_net[0], inputs = (im_input, ))
    # print('------- FLOPs: {:.3f} G'.format(flops/1000**3), '------- Model Size: {:.3f} MB'.format(model_size/1024**2))

    Phi_T = Phi[:, 0:Phi.shape[0], 0].unsqueeze(0).unsqueeze(0)

    mask_new = torch.zeros((Phi.shape[0], Phi.shape[1]))
    Phi_new = torch.unsqueeze(mask_new, 2).repeat(1, 1, Phi.shape[2]).to(device)

    step = args.step
    n_channels = Phi_new.shape[2]  # 28
    h, w = Phi_new.shape[0], Phi_new.shape[0]

    cols = torch.arange(w, device=device) + torch.arange(n_channels, device=device)[:,
                                            None] * step  # (28, 256)
    cols = cols.unsqueeze(0)  # 变为 (1, 28, 256)

    rows = torch.arange(h, device=device)[:, None, None]  # (256, 1, 1)

    channels = torch.arange(n_channels, device=device)[None, :, None]  # (1, 28, 1)

    for idx in range(iter_num):
        im_input_perturbed = im_input + im_input_temp.normal_() * 0.033
        model_out = im_net[0](im_input_perturbed)
        out_k = net_kernel(net_input_kernel).float().contiguous()
        out_k = out_k.view(1, 5)

        max_translation = args.max_translation
        max_angle = args.max_angle * math.pi / 180
        max_scale = args.max_scale
        h, w = Phi_T.shape[2], Phi_T.shape[3]
        scale_x = 1.0 + out_k[:, 0] * max_scale
        # scale_y = 1.0 + out_k[:, 1] * max_scale
        scale_y = scale_x
        angle = out_k[:, 2] * max_angle
        tx = out_k[:, 3] * 2 * max_translation / (w - 1)
        ty = out_k[:, 4] * 2 * max_translation / (h - 1)

        theta = torch.zeros(1, 2, 3, device=out_k.device, dtype=torch.float32)
        theta[:, 0, 0] = torch.cos(angle) * scale_x
        theta[:, 0, 1] = -torch.sin(angle) * scale_y
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = torch.sin(angle) * scale_x
        theta[:, 1, 1] = torch.cos(angle) * scale_y
        theta[:, 1, 2] = ty


        grid = F.affine_grid(theta, size=Phi_T.size(), align_corners=False)
        Phi_ = F.grid_sample(Phi_T, grid, align_corners=False)
        Phi_ = Phi_.squeeze()

        Phi_expanded = Phi_[:, :, None].to(device)
        Phi_new_updated = Phi_new.clone()
        Phi_new_updated[rows, cols, channels] = Phi_expanded.permute(0, 2, 1)
        pred_meas = A(shift(model_out.squeeze(0).permute(1, 2, 0), args.step).to(device), Phi_new_updated.to(device))

        if z == None:
            loss = args.lambda_R * loss_l1(meas, pred_meas)  # + (1/2)*loss_l2(meas, pred_meas)
        else:
            loss = args.lambda_R * loss_l1(meas, pred_meas)  # + (args.lambda_/2)*loss_l2(z, model_out)

        loss_tv = loss_l1(im_input, torch.zeros_like(im_input))
        loss += args.lambda_ * loss_tv


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_hs_recon = model_out.detach()

        if (idx + 1) % 10 == 0:
            video_gen(model_out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
            PSNR = calculate_psnr_tensor(truth_tensor, model_out.squeeze(0))
            print('Iter {}, x_loss:{:.3f}, s_loss:{:.3f}, PSNR:{:.2f}'.format(
                idx + 1, loss.item(), loss_tv.item(), PSNR))
            # print(ss_tv_loss(model_out))
            print(theta)
            mask_to_save = Phi_.detach().cpu().numpy()
            sio.savemat(os.path.join("Results/", f"mask.mat"),{'mask': mask_to_save})
            mask3d_to_save = Phi_new_updated.detach().cpu().numpy()
            sio.savemat(os.path.join("Results/", f"mask_3d_shift.mat"),{'mask': mask3d_to_save})
    end_time = time.time()

    # print('-------------- Finished----------, running time {:.1f} seconds.'.format(end_time - begin_time))
    return best_hs_recon.squeeze(0).permute(1, 2, 0)


def MK_LCTC(meas, Phi, z, truth_tensor, im_input, net_input_kernel, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    iter_num = args.LR_iter
    _, B, _, _ = truth_tensor.shape

    loss_l1 = torch.nn.L1Loss().to(device)

    base_im_net = CTC_model_load(args.ip_BI, B)
    base_net_kernel = fcn(5, 5)
    base_net_kernel = base_net_kernel.type(torch.cuda.FloatTensor)

    # load pretrained if exist
    save_model_weight = False if args.iter_num == 1 else True
    if os.path.exists('Results/model_weights.pth'):
        base_im_net[0].load_state_dict(torch.load('Results/model_weights.pth'))
        print('----------------------- Load model weights -----------------------')
        iter_num, save_model_weight = args.R_iter, True

    if os.path.exists('Results/kernel_weights.pth'):
        base_net_kernel.load_state_dict(torch.load('Results/kernel_weights.pth'))
        print('----------------------- Load kernel weights -----------------------')


    branch_count = args.kernel_num
    if args.add_order == 'forward':
        signs = [-0.5, 0.5]
    else:
        signs = [0.5, -0.5]
    branch_shifts = []
    for a in signs:
        for tx in signs:
            for ty in signs:
                b = torch.zeros(5, device=device, dtype=torch.float32)
                b[2] = a  # angle
                b[3] = tx  # tx
                b[4] = ty  # ty
                branch_shifts.append(b)

    im_nets, net_kernels, branch_bias_params, im_inputs, optimizers = [], [], [], [], []

    for i in range(branch_count):
        im_net_b = copy.deepcopy(base_im_net[0]).to(device)
        im_net_b.train()
        im_nets.append(im_net_b)

        net_kernel_b = copy.deepcopy(base_net_kernel).to(device)
        net_kernel_b.train()
        net_kernels.append(net_kernel_b)

        im_input_b = im_input.clone().detach().requires_grad_(True).to(device)
        im_inputs.append(im_input_b)

        bias_init = branch_shifts[i].clone().detach()
        bias_param = torch.nn.Parameter(bias_init)
        branch_bias_params.append(bias_param)

        net_params = list(im_net_b.parameters())
        kernel_params = list(net_kernel_b.parameters())
        input_params = [im_input_b]
        bias_params = [bias_param]

        opt = torch.optim.Adam([
            {'params': net_params, 'lr': 1e-3},
            {'params': input_params, 'lr': 1e-3},
            {'params': kernel_params, 'lr': 1e-3},
            {'params': bias_params, 'lr': 1e-3},
        ])
        optimizers.append(opt)

    Phi_T = Phi[:, 0:Phi.shape[0], 0].unsqueeze(0).unsqueeze(0)
    mask_new = torch.zeros((Phi.shape[0], Phi.shape[1]))
    Phi_new = torch.unsqueeze(mask_new, 2).repeat(1, 1, Phi.shape[2]).to(device)

    step = args.step
    n_channels = Phi_new.shape[2]
    h, w = Phi_new.shape[0], Phi_new.shape[0]

    cols = torch.arange(w, device=device) + torch.arange(n_channels, device=device)[:, None] * step
    cols = cols.unsqueeze(0)
    rows = torch.arange(h, device=device)[:, None, None]
    channels = torch.arange(n_channels, device=device)[None, :, None]

    loss_b = [float('inf')] * branch_count
    hs_recon_b = [None] * branch_count
    ss_tv_b = [float('inf')] * branch_count
    theta_b = [float('inf')] * branch_count

    begin_time = time.time()
    exploration_iters = min(args.switch_iters, iter_num)

    max_translation = args.max_translation
    max_angle = args.max_angle * math.pi / 180
    max_scale = args.max_scale

    for idx in range(exploration_iters):
        for b in range(branch_count):
            im_net_b = im_nets[b]
            net_kernel_b = net_kernels[b]
            im_input_b = im_inputs[b]
            opt_b = optimizers[b]
            bias_b = branch_bias_params[b]

            noise = torch.randn_like(im_input_b) * 0.033
            im_input_perturbed = im_input_b + noise

            model_out = im_net_b(im_input_perturbed)
            out_k = net_kernel_b(net_input_kernel).float().contiguous().view(1, 5)
            out_k = out_k + bias_b.view(1, 5)


            hT, wT = Phi_T.shape[2], Phi_T.shape[3]
            scale_x = 1.0 + out_k[:, 0] * max_scale
            # scale_y = 1.0 + out_k[:, 1] * max_scale
            scale_y = scale_x
            angle = out_k[:, 2] * max_angle
            tx = out_k[:, 3] * 2 * max_translation / (wT - 1)
            ty = out_k[:, 4] * 2 * max_translation / (hT - 1)

            theta = torch.zeros(1, 2, 3, device=out_k.device, dtype=torch.float32)
            theta[:, 0, 0] = torch.cos(angle) * scale_x
            theta[:, 0, 1] = -torch.sin(angle) * scale_y
            theta[:, 0, 2] = tx
            theta[:, 1, 0] = torch.sin(angle) * scale_x
            theta[:, 1, 1] = torch.cos(angle) * scale_y
            theta[:, 1, 2] = ty
            theta_b[b] = theta

            grid = F.affine_grid(theta, size=Phi_T.size(), align_corners=False)
            Phi_ = F.grid_sample(Phi_T, grid, align_corners=False).squeeze()

            Phi_expanded = Phi_[:, :, None].to(device)
            Phi_new_updated = Phi_new.clone()
            Phi_new_updated[rows, cols, channels] = Phi_expanded.permute(0, 2, 1)

            pred_meas = A(shift(model_out.squeeze(0).permute(1, 2, 0), args.step).to(device),
                          Phi_new_updated.to(device))

            loss = args.lambda_R * loss_l1(meas, pred_meas)
            loss_tv = loss_l1(im_input_b, torch.zeros_like(im_input_b))
            loss = loss + args.lambda_ * loss_tv

            opt_b.zero_grad()
            loss.backward()
            opt_b.step()

            with torch.no_grad():
                ss_tv_val = ss_tv_loss(model_out).item()
                ss_tv_b[b] = ss_tv_val
                hs_recon_b[b] = model_out.detach().clone()


        if (idx + 1) % 100 == 0:
            print(f'Exploration iter {idx+1}/{exploration_iters}')

    best_ss_tv_tensor = torch.tensor(ss_tv_b, device='cpu', dtype=torch.float32)
    choose_idx = int(torch.argmin(best_ss_tv_tensor).item())
    print(f'-- After exploration choose branch {choose_idx} with ss_tv={ss_tv_b[choose_idx]:.6f} and data-loss={loss_b[choose_idx]:.4f}')

    remaining_iters = iter_num - exploration_iters
    print(f'-- Fine-tuning only branch {choose_idx} for {remaining_iters} more iterations.')

    opt_chosen = optimizers[choose_idx]
    im_net_ch = im_nets[choose_idx]
    net_kernel_ch = net_kernels[choose_idx]
    im_input_ch = im_inputs[choose_idx]
    bias_ch = branch_bias_params[choose_idx]

    for idx2 in range(remaining_iters):
        noise = torch.randn_like(im_input_ch) * 0.033
        im_input_perturbed = im_input_ch + noise

        model_out = im_net_ch(im_input_perturbed)
        out_k = net_kernel_ch(net_input_kernel).float().contiguous().view(1, 5)
        out_k = out_k + bias_ch.view(1, 5)

        hT, wT = Phi_T.shape[2], Phi_T.shape[3]
        scale_x = 1.0 + out_k[:, 0] * max_scale
        scale_y = scale_x
        angle = out_k[:, 2] * max_angle
        tx = out_k[:, 3] * 2 * max_translation / (wT - 1)
        ty = out_k[:, 4] * 2 * max_translation / (hT - 1)


        theta = torch.zeros(1, 2, 3, device=out_k.device, dtype=torch.float32)
        theta[:, 0, 0] = torch.cos(angle) * scale_x
        theta[:, 0, 1] = -torch.sin(angle) * scale_y
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = torch.sin(angle) * scale_x
        theta[:, 1, 1] = torch.cos(angle) * scale_y
        theta[:, 1, 2] = ty
        theta_b[choose_idx] = theta
        grid = F.affine_grid(theta, size=Phi_T.size(), align_corners=False)
        Phi_ = F.grid_sample(Phi_T, grid, align_corners=False).squeeze()

        Phi_expanded = Phi_[:, :, None].to(device)
        Phi_new_updated = Phi_new.clone()
        Phi_new_updated[rows, cols, channels] = Phi_expanded.permute(0, 2, 1)

        pred_meas = A(shift(model_out.squeeze(0).permute(1, 2, 0), args.step).to(device),
                      Phi_new_updated.to(device))

        loss = args.lambda_R * loss_l1(meas, pred_meas)
        loss_tv = loss_l1(im_input_ch, torch.zeros_like(im_input_ch))
        loss = loss + args.lambda_ * loss_tv

        opt_chosen.zero_grad()
        loss.backward()
        opt_chosen.step()

        with torch.no_grad():
            ss_tv_val = ss_tv_loss(model_out).item()
            loss_b[choose_idx] = loss.item()
            hs_recon_b[choose_idx] = model_out.detach().clone()
            ss_tv_b[choose_idx] = ss_tv_val

        if (idx2 + 1) % 100 == 0:
            print(theta.detach().cpu().numpy())
            PSNR_ch = calculate_psnr_tensor(truth_tensor, hs_recon_b[choose_idx].squeeze(0))
            print(f'FineTune iter {idx2+1}/{remaining_iters} - Branch {choose_idx}: loss={loss_b[choose_idx]:.4f}, ss_tv={ss_tv_b[choose_idx]:.6f}, PSNR={PSNR_ch:.2f}')

    end_time = time.time()
    print('-------------- Finished, running time {:.1f} seconds.'.format(end_time - begin_time))

    best_hs_recon = hs_recon_b[choose_idx]
    return best_hs_recon.squeeze(0).permute(1, 2, 0)



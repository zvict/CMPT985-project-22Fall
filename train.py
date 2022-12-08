import yaml
import argparse
import torch
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from dataset import get_traindataset, get_trainloader, get_testdataset
from models import get_model, get_loss
from dataset.utils import cam_to_world, world_to_cam


class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="PAPR")
    parser.add_argument('--opt', type=str, default="", help='Option file path')
    parser.add_argument('--resume', type=int, default=0, help='job array id')
    return parser.parse_args()


def eval_step(step, model, device, dataset, eval_dataset, batch, loss_fn, train_out, args, train_losses, eval_losses, eval_psnrs, steps):
    train_img_idx, _, _, _, _, train_patch = batch
    train_img, _, _, train_norm = dataset.get_full_img(train_img_idx[0])
    img, rayd, rayo, norm = eval_dataset.get_full_img(args.eval.img_idx)
    c2w = dataset.get_c2w(args.eval.img_idx)

    # print("Eval step:", step, "img_idx:", args.eval.img_idx)
    
    _, H, W, _ = rayd.shape
    num_pts, _ = model.points.shape
    pt_idxs = args.geoms.points.eval_idxs

    rayo = rayo.to(device)
    rayd = rayd.to(device)
    norm = norm.to(device)
    c2w = c2w.to(device)

    # print("Eval step:", step, "img_idx:", args.eval.img_idx)

    # encode = torch.zeros(1, H, W, num_pts, args.models.transformer.ff_dim)
    vnorm = torch.zeros(1, H, W, num_pts, 3)
    attn = torch.zeros(1, H, W, num_pts, 1)
    out = torch.zeros_like(norm)
    # first_layer_attn = torch.zeros(model.n_kernel, H, W, num_pts, num_pts)
    # first_layer_out = torch.zeros(H, W, num_pts, args.models.transformer.ff_dim)
    selected_first_layer_attn = torch.zeros(model.n_kernel, num_pts, num_pts)
    selected_first_layer_out = torch.zeros(H, W, len(pt_idxs), args.models.transformer.ff_dim)
    num_layers = args.models.transformer.num_layers - 1
    num_heads = args.models.transformer.num_heads
    if num_layers > 0:
        # encoder_layer_attn = torch.zeros(num_layers, H, W, num_heads, num_pts, num_pts)
        # encoder_layer_out = torch.zeros(num_layers, H, W, num_pts, args.models.transformer.ff_dim)
        selected_encoder_layer_attn = torch.zeros(num_layers, num_heads, num_pts, num_pts)
        selected_encoder_layer_out = torch.zeros(num_layers, H, W, len(pt_idxs), args.models.transformer.ff_dim)

    # print("Eval step:", step, "img_idx:", args.eval.img_idx)

    num_heads_1st_layer = getattr(model.transformer, "selfattn_1").h

    # model.eval()
    with torch.no_grad():
        for height_start in range(0, H, args.eval.max_height):
            for width_start in range(0, W, args.eval.max_width):
                height_end = min(height_start + args.eval.max_height, H)
                width_end = min(width_start + args.eval.max_width, W)

                # print("Eval step:", step, "img_idx:", args.eval.img_idx, "height_start:", height_start, "width_start:", width_start)

                _, \
                vnorm[:, height_start:height_end, width_start:width_end, :, :], \
                attn[:, height_start:height_end, width_start:width_end, :, :], \
                out[:, height_start:height_end, width_start:width_end] = model.evaluate(rayo, rayd[:, height_start:height_end, width_start:width_end], c2w)

                # print(model.transformer.first_layer.self_attn.attn.shape)
                # print(model.transformer.encoder.layers[0].self_attn.attn.shape)
                # print(model.transformer.encoder.layers[1].self_attn.attn.shape)
                # print(model.transformer.encoder.layers[2].self_attn.attn.shape)
                # print(model.transformer.first_layer.middle_outputs[0].shape)
                # print(model.transformer.encoder.middle_outputs[0].shape)
                # print(model.transformer.encoder.middle_outputs[1].shape)
                # print(model.transformer.encoder.middle_outputs[2].shape)

                # print("Eval step:", step, "img_idx:", args.eval.img_idx, "height_start:", height_start, "width_start:", width_start)

                cur_h, cur_w = height_end - height_start, width_end - width_start
                if (H // 2) >= height_start and (H // 2) < height_end and (W // 2) >= width_start and (W // 2) < width_end:
                    selected_first_layer_attn = getattr(model.transformer, "selfattn_1").attn.reshape(num_heads_1st_layer, cur_h, cur_w, num_pts, num_pts)[:, H // 2 - height_start, W // 2 - width_start]
                    if num_layers > 0:
                        for i in range(num_layers):
                            selected_encoder_layer_attn[i] = getattr(model.transformer, f"selfattn_{i+2}").attn.reshape(cur_h, cur_w, num_heads, num_pts, num_pts)[H // 2 - height_start, W // 2 - width_start]

                selected_first_layer_out[height_start:height_end, width_start:width_end] = model.transformer.middle_outputs[0].reshape(cur_h, cur_w, num_pts, -1)[:, :, pt_idxs, :]
                if num_layers > 0:
                    for i in range(num_layers):
                        selected_encoder_layer_out[i, height_start:height_end, width_start:width_end] = model.transformer.middle_outputs[i+1].reshape(cur_h, cur_w, num_pts, -1)[:, :, pt_idxs, :]

        eval_loss = loss_fn(out, norm)
        eval_psnr = -10. * np.log(((out - norm)**2).mean().item()) / np.log(10.)

    # print("Eval step:", step, "img_idx:", args.eval.img_idx)

    # selected_first_layer_attn = first_layer_attn[:, H // 2, W // 2, :, :].detach().cpu().numpy()    # (n_kernel, num_pts, num_pts)
    # selected_first_layer_out = first_layer_out[:, :, pt_idxs, :].detach().cpu().numpy()   # (H, W, selected_pts, ff_dim)
    # del first_layer_attn, first_layer_out
    # if num_layers > 0:
    #     selected_encoder_layer_attn = encoder_layer_attn[:, H // 2, W // 2, :, :, :].detach().cpu().numpy()   # (num_layers, num_heads, num_pts, num_pts)
    #     selected_encoder_layer_out = encoder_layer_out[:, :, :, pt_idxs, :].detach().cpu().numpy()    # (num_layers, H, W, selected_pts, ff_dim)
    #     del encoder_layer_attn, encoder_layer_out

    selected_first_layer_attn = selected_first_layer_attn.detach().cpu().numpy()
    selected_first_layer_out = selected_first_layer_out.detach().cpu().numpy()
    if num_layers > 0:
        selected_encoder_layer_attn = selected_encoder_layer_attn.detach().cpu().numpy()
        selected_encoder_layer_out = selected_encoder_layer_out.detach().cpu().numpy()

    # steps.append(step)
    eval_losses.append(eval_loss.item())
    eval_psnrs.append(eval_psnr)

    print("Eval step:", step, "train_loss:", train_losses[-1], "eval_loss:", eval_losses[-1], "eval_psnr:", eval_psnrs[-1])

    log_dir = os.path.join("experiments", args.index)
    os.makedirs(os.path.join(log_dir, "figs"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "middles"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "ptattns"), exist_ok=True)

    if args.models.norm_mlp.out_type == 'norm':
        train_norm = train_norm / 2 + 0.5
        train_patch = train_patch / 2 + 0.5
        train_out = train_out / 2 + 0.5
        norm = norm / 2 + 0.5
        out = out / 2 + 0.5

    if args.eval.save_fig:
        # main plot
        fig = plt.figure(figsize=(20, 10))

        ax = fig.add_subplot(2, 5, 1)
        ax.imshow(train_norm.squeeze().cpu().numpy().astype(np.float32))
        ax.set_title(f'Iteration: {step} train norm')

        ax = fig.add_subplot(2, 5, 2)
        ax.imshow(train_patch[0].cpu().numpy().astype(np.float32))
        ax.set_title(f'Iteration: {step} train norm patch')

        ax = fig.add_subplot(2, 5, 3)
        ax.imshow(train_out[0])
        ax.set_title(f'Iteration: {step} train output')

        ax = fig.add_subplot(2, 5, 4)
        ax.plot(steps, train_losses)
        ax.set_title('train loss')

        points_np = model.points.detach().cpu().numpy()
        norms_np = model.pc_norms.detach().cpu().numpy()
        ax = fig.add_subplot(2, 5, 5, projection='3d')
        ax.set_xlim3d(-0.8, 0.8)
        ax.set_ylim3d(-0.8, 0.8)
        ax.set_zlim3d(-0.8, 0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], color="grey")
        for i in range(points_np.shape[0]):
            ax.quiver(points_np[i, 0], points_np[i, 1], points_np[i, 2], norms_np[i, 0], norms_np[i, 1], norms_np[i, 2], length=0.2, alpha=1, color="blue")
        ax.set_title('Point Cloud')

        # ax = fig.add_subplot(2, 5, 6)
        # ax.imshow(img.squeeze().cpu().numpy().astype(np.float32))
        # ax.set_title(f'Iteration: {step} eval img')

        ax = fig.add_subplot(2, 5, 7)
        ax.imshow(norm.squeeze().cpu().numpy().astype(np.float32))
        ax.set_title(f'Iteration: {step} eval norm')

        ax = fig.add_subplot(2, 5, 8)
        ax.imshow(out.squeeze().detach().cpu().numpy().astype(np.float32))
        ax.set_title(f'Iteration: {step} eval predict')

        ax = fig.add_subplot(2, 5, 9)
        ax.plot(steps, eval_losses)
        ax.set_title('eval loss')

        ax = fig.add_subplot(2, 5, 10)
        ax.plot(steps, eval_psnrs)
        ax.set_title('eval psnr')

        fig.suptitle("iter %d" % (step))
        save_name = os.path.join(log_dir, "figs", "%s_iter_%d.png" % (args.index, step))
        fig.savefig(save_name)
        plt.close()

        # middle output plot
        num_rows = num_layers + 2
        fig = plt.figure(figsize=(24, 5 * num_rows))

        if args.models.norm_mlp.out_type == 'norm':
            vnorm = vnorm.squeeze().detach().cpu().numpy() / 2. + 0.5
        elif args.models.norm_mlp.out_type == 'rgb':
            vnorm = vnorm.squeeze().detach().cpu().numpy()
        else:
            raise NotImplementedError
        attn = attn.squeeze().detach().cpu().numpy()

        ax = fig.add_subplot(num_rows, 6, 1)
        cb = ax.imshow(vnorm[:, :, pt_idxs[0], :])
        ax.set_title(f'Eval vnorm point {pt_idxs[0]}')

        ax = fig.add_subplot(num_rows, 6, 2)
        cb = ax.imshow(vnorm[:, :, pt_idxs[1], :])
        ax.set_title(f'Eval vnorm point {pt_idxs[1]}')

        ax = fig.add_subplot(num_rows, 6, 3)
        cb = ax.imshow(vnorm[:, :, pt_idxs[2], :])
        ax.set_title(f'Eval vnorm point {pt_idxs[2]}')

        ax = fig.add_subplot(num_rows, 6, 4)
        minv = attn[:, :, pt_idxs[0]].min()
        maxv = attn[:, :, pt_idxs[0]].max()
        cb = ax.imshow((attn[:, :, pt_idxs[0]] - minv) / (maxv - minv))
        ax.set_title(f'Eval attn point {pt_idxs[0]}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(num_rows, 6, 5)
        minv = attn[:, :, pt_idxs[1]].min()
        maxv = attn[:, :, pt_idxs[1]].max()
        cb = ax.imshow((attn[:, :, pt_idxs[1]] - minv) / (maxv - minv))
        ax.set_title(f'Eval attn point {pt_idxs[1]}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(num_rows, 6, 6)
        minv = attn[:, :, pt_idxs[2]].min()
        maxv = attn[:, :, pt_idxs[2]].max()
        cb = ax.imshow((attn[:, :, pt_idxs[2]] - minv) / (maxv - minv))
        ax.set_title(f'Eval attn point {pt_idxs[2]}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(num_rows, 6, 7)
        kernel_idx = 0 if num_heads_1st_layer >= 1 else 0
        minv = selected_first_layer_attn[kernel_idx, :, :].min()
        maxv = selected_first_layer_attn[kernel_idx, :, :].max()
        cb = ax.imshow((selected_first_layer_attn[kernel_idx, :, :] - minv) / (maxv - minv))
        ax.set_title(f'Eval 1st layer attn kernel {kernel_idx} centre ray\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(num_rows, 6, 8)
        kernel_idx = 1 if num_heads_1st_layer >= 2 else 0
        minv = selected_first_layer_attn[kernel_idx, :, :].min()
        maxv = selected_first_layer_attn[kernel_idx, :, :].max()
        cb = ax.imshow((selected_first_layer_attn[kernel_idx, :, :] - minv) / (maxv - minv))
        ax.set_title(f'Eval 1st layer attn kernel {kernel_idx} centre ray\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(num_rows, 6, 9)
        kernel_idx = 2 if num_heads_1st_layer >= 3 else 0
        minv = selected_first_layer_attn[kernel_idx, :, :].min()
        maxv = selected_first_layer_attn[kernel_idx, :, :].max()
        cb = ax.imshow((selected_first_layer_attn[kernel_idx, :, :] - minv) / (maxv - minv))
        ax.set_title(f'Eval 1st layer attn kernel {kernel_idx} centre ray\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(num_rows, 6, 10)
        ff_idx = args.models.transformer.ff_dim // 2
        minv = selected_first_layer_out[:, :, 0, ff_idx].min()
        maxv = selected_first_layer_out[:, :, 0, ff_idx].max()
        cb = ax.imshow((selected_first_layer_out[:, :, 0, ff_idx] - minv) / (maxv - minv))
        ax.set_title(f'Eval 1st layer out pt {pt_idxs[0]} dim {ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(num_rows, 6, 11)
        ff_idx = args.models.transformer.ff_dim // 2
        minv = selected_first_layer_out[:, :, 1, ff_idx].min()
        maxv = selected_first_layer_out[:, :, 1, ff_idx].max()
        cb = ax.imshow((selected_first_layer_out[:, :, 1, ff_idx] - minv) / (maxv - minv))
        ax.set_title(f'Eval 1st layer out pt {pt_idxs[1]} dim {ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(num_rows, 6, 12)
        ff_idx = args.models.transformer.ff_dim // 2
        minv = selected_first_layer_out[:, :, 2, ff_idx].min()
        maxv = selected_first_layer_out[:, :, 2, ff_idx].max()
        cb = ax.imshow((selected_first_layer_out[:, :, 2, ff_idx] - minv) / (maxv - minv))
        ax.set_title(f'Eval 1st layer out pt {pt_idxs[2]} dim {ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        for i in range(num_layers):
            ax = fig.add_subplot(num_rows, 6, 13 + i * 6)
            head_idx = 0 if num_heads >= 1 else 0
            minv = selected_encoder_layer_attn[i, head_idx, :, :].min()
            maxv = selected_encoder_layer_attn[i, head_idx, :, :].max()
            cb = ax.imshow((selected_encoder_layer_attn[i, head_idx, :, :] - minv) / (maxv - minv))
            ax.set_title(f'Eval layer {i+2} attn head {head_idx} centre ray\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

            ax = fig.add_subplot(num_rows, 6, 14 + i * 6)
            head_idx = 1 if num_heads >= 2 else 0
            minv = selected_encoder_layer_attn[i, head_idx, :, :].min()
            maxv = selected_encoder_layer_attn[i, head_idx, :, :].max()
            cb = ax.imshow((selected_encoder_layer_attn[i, head_idx, :, :] - minv) / (maxv - minv))
            ax.set_title(f'Eval layer {i+2} attn head {head_idx} centre ray\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

            ax = fig.add_subplot(num_rows, 6, 15 + i * 6)
            head_idx = 2 if num_heads >= 3 else 0
            minv = selected_encoder_layer_attn[i, head_idx, :, :].min()
            maxv = selected_encoder_layer_attn[i, head_idx, :, :].max()
            cb = ax.imshow((selected_encoder_layer_attn[i, head_idx, :, :] - minv) / (maxv - minv))
            ax.set_title(f'Eval layer {i+2} attn head {head_idx} centre ray\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

            ax = fig.add_subplot(num_rows, 6, 16 + i * 6)
            ff_idx = args.models.transformer.ff_dim // 2
            minv = selected_encoder_layer_out[i, :, :, 0, ff_idx].min()
            maxv = selected_encoder_layer_out[i, :, :, 0, ff_idx].max()
            cb = ax.imshow((selected_encoder_layer_out[i, :, :, 0, ff_idx] - minv) / (maxv - minv))
            ax.set_title(f'Eval layer {i+2} out pt {pt_idxs[0]} dim {ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

            ax = fig.add_subplot(num_rows, 6, 17 + i * 6)
            ff_idx = args.models.transformer.ff_dim // 2
            minv = selected_encoder_layer_out[i, :, :, 1, ff_idx].min()
            maxv = selected_encoder_layer_out[i, :, :, 1, ff_idx].max()
            cb = ax.imshow((selected_encoder_layer_out[i, :, :, 1, ff_idx] - minv) / (maxv - minv))
            ax.set_title(f'Eval layer {i+2} out pt {pt_idxs[1]} dim {ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

            ax = fig.add_subplot(num_rows, 6, 18 + i * 6)
            ff_idx = args.models.transformer.ff_dim // 2
            minv = selected_encoder_layer_out[i, :, :, 2, ff_idx].min()
            maxv = selected_encoder_layer_out[i, :, :, 2, ff_idx].max()
            cb = ax.imshow((selected_encoder_layer_out[i, :, :, 2, ff_idx] - minv) / (maxv - minv))
            ax.set_title(f'Eval layer {i+2} out pt {pt_idxs[2]} dim {ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))        

        fig.suptitle("iter %d" % (step))
        save_name = os.path.join(log_dir, "middles", "%s_iter_%d.png" % (args.index, step))
        fig.savefig(save_name)
        plt.close()

        # point attention plot
        num_rows = num_layers + 1
        fig = plt.figure(figsize=(20, 4 * num_rows))

        # rayo = rayo.cpu().numpy()
        # rayd = rayd[H // 2, W // 2].cpu().numpy()

        def get_colors(weights):
            N = weights.shape[0]
            weights = (weights - weights.min()) / (weights.max() - weights.min())
            colors = np.full((N, 3), [1., 0., 0.])
            colors[:, 0] *= weights[:N]
            colors[:, 2] = (1 - weights[:N])
            return colors

        for i, pt_idx in enumerate(pt_idxs):
            ax = fig.add_subplot(num_rows, 5, i+1, projection='3d')
            ax.set_xlim3d(-0.8, 0.8)
            ax.set_ylim3d(-0.8, 0.8)
            ax.set_zlim3d(-0.8, 0.8)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=get_colors(selected_first_layer_attn[0, pt_idx, :]), s=1)
            ax.scatter(points_np[pt_idx, 0], points_np[pt_idx, 1], points_np[pt_idx, 2], c='black', marker='*', s=2)
            ax.set_title(f'First layer pt {pt_idx} attn center ray H0')

        for l in range(num_layers):
            for i, pt_idx in enumerate(pt_idxs):
                ax = fig.add_subplot(num_rows, 5, (l+1)*5+i+1, projection='3d')
                ax.set_xlim3d(-0.8, 0.8)
                ax.set_ylim3d(-0.8, 0.8)
                ax.set_zlim3d(-0.8, 0.8)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=get_colors(selected_encoder_layer_attn[l, 0, pt_idx, :]), s=1)
                ax.scatter(points_np[pt_idx, 0], points_np[pt_idx, 1], points_np[pt_idx, 2], c='black', marker='*', s=10)
                ax.set_title(f'{l} layer pt {pt_idx} attn center ray H0')

        fig.suptitle("iter %d" % (step))
        save_name = os.path.join(log_dir, "ptattns", "%s_iter_%d.png" % (args.index, step))
        fig.savefig(save_name)
        plt.close()

    model.save(step, log_dir)
    if step % 50000 == 0:
        torch.save(model.state_dict(), os.path.join(log_dir, "model_%d.pth" % step))

    torch.save(torch.tensor(train_losses), os.path.join(log_dir, "train_losses.pth"))
    torch.save(torch.tensor(eval_losses), os.path.join(log_dir, "eval_losses.pth"))
    torch.save(torch.tensor(eval_psnrs), os.path.join(log_dir, "eval_psnrs.pth"))


def train_step(step, model, device, dataset, batch, loss_fn, args):
    img_idx, patch_idx, img, rayd, rayo, norm = batch
    c2w = dataset.get_c2w(img_idx[0])

    rayo = rayo.to(device)
    rayd = rayd.to(device)
    norm = norm.to(device)
    c2w = c2w.to(device)

    model.train()
    model.clear_grad()
    out = model(rayo, rayd, c2w, step)
    loss = loss_fn(out, norm)
    loss.backward()
    model.step()

    return loss.item(), out.detach().cpu().numpy()


def train_and_eval(start_step, model, device, dataset, eval_dataset, losses, args):
    trainloader = get_trainloader(dataset, args.dataset)
    print("trainloader:", trainloader)

    loss_fn = get_loss(args.training.losses)
    loss_fn = loss_fn.to(device)

    # steps = []
    # train_losses = []
    # eval_losses = []
    # eval_psnrs = []
    train_losses, eval_losses, eval_psnrs = losses

    avg_train_loss = 0.
    step = start_step
    print(start_step, args.training.steps, args.training.chunk_steps)
    while step < args.training.steps and (step - start_step) < args.training.chunk_steps:
        for i, batch in enumerate(trainloader):

            loss, out = train_step(step, model, device, dataset, batch, loss_fn, args)
            avg_train_loss += loss
            step += 1
            
            if step % 20 == 0:
                print("Train step:", step, "loss:", loss, "tx_lr:", model.tx_lr)

            if step % args.eval.step == 0:
                train_losses.append(avg_train_loss / args.eval.step)
                steps = [(s + 1) * args.eval.step for s in range(len(train_losses))]
                eval_step(step, model, device, dataset, eval_dataset, batch, loss_fn, out, args, train_losses, eval_losses, eval_psnrs, steps)
                avg_train_loss = 0.

            if step >= args.training.steps or (step - start_step) >= args.training.chunk_steps:
                break

            
def main(args, eval_args, resume):
    model = get_model(args)
    dataset = get_traindataset(args.dataset)
    eval_dataset = get_testdataset(eval_args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    start_step = 0
    losses = [[], [], []]
    if resume > 0:
        log_dir = os.path.join("experiments", args.index)
        start_step = model.load(log_dir)

        train_losses = torch.load(os.path.join(log_dir, "train_losses.pth")).tolist()
        eval_losses = torch.load(os.path.join(log_dir, "eval_losses.pth")).tolist()
        eval_psnrs = torch.load(os.path.join(log_dir, "eval_psnrs.pth")).tolist()
        losses = [train_losses, eval_losses, eval_psnrs]

    train_and_eval(start_step, model, device, dataset, eval_dataset, losses, args)


if __name__ == '__main__':

    args = parse_args()
    with open(args.opt, 'r') as f:
        config = yaml.safe_load(f)
    eval_config = copy.deepcopy(config)
    eval_config['dataset'].update(eval_config['eval']['dataset'])
    eval_config = DictAsMember(eval_config)
    config = DictAsMember(config)

    log_dir = os.path.join("experiments", config.index)
    os.makedirs(log_dir, exist_ok=True)

    shutil.copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))
    shutil.copyfile(args.opt, os.path.join(log_dir, os.path.basename(args.opt)))

    setup_seed(config.seed)

    # print(config.geoms.points.init_scale, config.geoms.points.init_center)

    main(config, eval_config, args.resume)

    # print(config.training.losses)
    # for loss, weight in config.training.losses.items():
    #     print(loss, weight, str(format(weight, '.1E')), float(str(format(weight, '.1E'))))
    # print(str(config.training.losses[1].values()[0]))
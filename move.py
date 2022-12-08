import yaml
import argparse
import torch
import os
import io
import shutil
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import numpy as np
import random
from dataset import get_testdataset, get_testloader
from models import get_model, get_loss


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
    return parser.parse_args()


def test_step(frame, num_frames, model, device, dataset, batch, loss_fn, args, test_losses, test_psnrs):
    idx, img, rayd, rayo, norm = batch
    c2w = dataset.get_c2w(idx.squeeze())

    _, H, W, _ = rayd.shape
    num_pts, _ = model.points.shape

    rayo = rayo.to(device)
    rayd = rayd.to(device)
    norm = norm.to(device)
    c2w = c2w.to(device)

    encode = torch.zeros(1, H, W, num_pts, args.models.transformer.ff_dim)
    vnorm = torch.zeros(1, H, W, num_pts, 3)
    attn = torch.zeros(1, H, W, num_pts, 1)
    out = torch.zeros_like(norm)
    # model.eval()
    with torch.no_grad():
        for height_start in range(0, H, args.test.max_height):
            for width_start in range(0, W, args.test.max_width):
                height_end = min(height_start + args.test.max_height, H)
                width_end = min(width_start + args.test.max_width, W)
                encode[:, height_start:height_end, width_start:width_end, :, :], \
                vnorm[:, height_start:height_end, width_start:width_end, :, :], \
                attn[:, height_start:height_end, width_start:width_end, :, :], \
                out[:, height_start:height_end, width_start:width_end] = model.evaluate(rayo, rayd[:, height_start:height_end, width_start:width_end], c2w)
        test_loss = loss_fn(out, norm)
        test_psnr = -10. * np.log(((out - norm)**2).mean().item()) / np.log(10.)

    test_losses.append(test_loss.item())
    test_psnrs.append(test_psnr)

    print("Test frame:", frame, "test_loss:", test_losses[-1], "test_psnr:", test_psnrs[-1], "out:", out.min().item(), out.max().item(), "norm:", norm.min().item(), norm.max().item())

    points_np = model.points.detach().cpu().numpy()
    norms_np = model.pc_norms.detach().cpu().numpy()

    plots = {}
    pt_idxs = args.geoms.points.eval_idxs
    # pt_idxs = [12, 32, 77, 85, 87]
    # pt_idxs = [12, 32, 66, 1, 68]

    if args.models.norm_mlp.out_type == "norm":
        out = out / 2 + 0.5
        norm = norm / 2 + 0.5
        vnorm = vnorm / 2 + 0.5

    plot_opt = args.test.plots
    th = -frame * (360. / num_frames)
    if plot_opt.pcrgb:
        fig = plt.figure(figsize=(30, 10))

        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.axis('off')
        ax.view_init(elev=20., azim=90 - th)
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter3D(points_np[:, 0], points_np[:, 1], points_np[:, 2], c='orange')

        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.axis('off')
        ax.view_init(elev=20., azim=90 - th)
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter3D(points_np[:, 0], points_np[:, 1], points_np[:, 2], c='orange')
        for i in range(num_pts):
            ax.text(points_np[i, 0], points_np[i, 1], points_np[i, 2], str(i))

        ax = fig.add_subplot(1, 3, 3)
        ax.axis('off')
        ax.imshow(out.squeeze().detach().cpu().numpy().astype(np.float32))

        fig.suptitle("frame %d, PSNR %.3f" % (frame, test_psnr))

        canvas = fig.canvas
        buffer = io.BytesIO()  # 获取输入输出流对象
        canvas.print_png(buffer)  # 将画布上的内容打印到输入输出流对象
        data = buffer.getvalue()  # 获取流的值
        buffer.write(data)  # 将数据写入buffer 
        img = Image.open(buffer)  # 使用Image打开图片数据
        plots["pcrgb"] = img
        plt.close()

    if plot_opt.vnormattn:
        fig = plt.figure(figsize=(20, 15))

        vnorm = vnorm.squeeze().detach().cpu().numpy()
        attn = attn.squeeze().detach().cpu().numpy()

        ax = fig.add_subplot(3, 5, 1, projection='3d')
        ax.axis('off')
        ax.view_init(elev=20., azim=90 - th)
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter3D(points_np[:, 0], points_np[:, 1], points_np[:, 2], c='orange')

        ax = fig.add_subplot(3, 5, 2)
        ax.axis('off')
        ax.imshow(norm.squeeze().cpu().numpy().astype(np.float32))

        ax = fig.add_subplot(3, 5, 3)
        ax.axis('off')
        ax.imshow(out.squeeze().detach().cpu().numpy().astype(np.float32))

        ax = fig.add_subplot(3, 5, 6)
        cb = ax.imshow(vnorm[:, :, pt_idxs[0], :] * 0.5 + 0.5)
        ax.set_title(f'Eval vnorm point {pt_idxs[0]}')

        ax = fig.add_subplot(3, 5, 7)
        cb = ax.imshow(vnorm[:, :, pt_idxs[1], :] * 0.5 + 0.5)
        ax.set_title(f'Eval vnorm point {pt_idxs[1]}')

        ax = fig.add_subplot(3, 5, 8)
        cb = ax.imshow(vnorm[:, :, pt_idxs[2], :] * 0.5 + 0.5)
        ax.set_title(f'Eval vnorm point {pt_idxs[2]}')

        ax = fig.add_subplot(3, 5, 9)
        cb = ax.imshow(vnorm[:, :, pt_idxs[3], :] * 0.5 + 0.5)
        ax.set_title(f'Eval vnorm point {pt_idxs[3]}')

        ax = fig.add_subplot(3, 5, 10)
        cb = ax.imshow(vnorm[:, :, pt_idxs[4], :] * 0.5 + 0.5)
        ax.set_title(f'Eval vnorm point {pt_idxs[4]}')

        ax = fig.add_subplot(3, 5, 11)
        minv = attn[:, :, pt_idxs[0]].min()
        maxv = attn[:, :, pt_idxs[0]].max()
        cb = ax.imshow((attn[:, :, pt_idxs[0]] - minv) / (maxv - minv))
        ax.set_title(f'Eval attn point {pt_idxs[0]}\n' + 'Real min: %.3f, max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(3, 5, 12)
        minv = attn[:, :, pt_idxs[1]].min()
        maxv = attn[:, :, pt_idxs[1]].max()
        cb = ax.imshow((attn[:, :, pt_idxs[1]] - minv) / (maxv - minv))
        ax.set_title(f'Eval attn point {pt_idxs[1]}\n' + 'Real min: %.3f, max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(3, 5, 13)
        minv = attn[:, :, pt_idxs[2]].min()
        maxv = attn[:, :, pt_idxs[2]].max()
        cb = ax.imshow((attn[:, :, pt_idxs[2]] - minv) / (maxv - minv))
        ax.set_title(f'Eval attn point {pt_idxs[2]}\n' + 'Real min: %.3f, max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(3, 5, 14)
        minv = attn[:, :, pt_idxs[3]].min()
        maxv = attn[:, :, pt_idxs[3]].max()
        cb = ax.imshow((attn[:, :, pt_idxs[3]] - minv) / (maxv - minv))
        ax.set_title(f'Eval attn point {pt_idxs[3]}\n' + 'Real min: %.3f, max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(3, 5, 15)
        minv = attn[:, :, pt_idxs[4]].min()
        maxv = attn[:, :, pt_idxs[4]].max()
        cb = ax.imshow((attn[:, :, pt_idxs[4]] - minv) / (maxv - minv))
        ax.set_title(f'Eval attn point {pt_idxs[4]}\n' + 'Real min: %.3f, max: %.3f' % (minv, maxv))

        fig.suptitle("frame %d\n12: room back, 32: roof, 77: air vent, 85: board front, 87 shovel front" % (frame))
        
        canvas = fig.canvas
        buffer = io.BytesIO()  # 获取输入输出流对象
        canvas.print_png(buffer)  # 将画布上的内容打印到输入输出流对象
        data = buffer.getvalue()  # 获取流的值
        buffer.write(data)  # 将数据写入buffer 
        img = Image.open(buffer)  # 使用Image打开图片数据
        plots["vnormattn"] = img
        plt.close()

    if plot_opt.txoutput:
        fig = plt.figure(figsize=(20, 15))

        encode = encode.squeeze().detach().cpu().numpy()

        ax = fig.add_subplot(3, 5, 1, projection='3d')
        ax.axis('off')
        ax.view_init(elev=20., azim=90 - th)
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter3D(points_np[:, 0], points_np[:, 1], points_np[:, 2], c='orange')

        ax = fig.add_subplot(3, 5, 2)
        ax.axis('off')
        ax.imshow(norm.squeeze().cpu().numpy().astype(np.float32))

        ax = fig.add_subplot(3, 5, 3)
        ax.axis('off')
        ax.imshow(out.squeeze().detach().cpu().numpy().astype(np.float32))

        ff_idx = args.models.transformer.ff_dim // 3
        ax = fig.add_subplot(3, 5, 6)
        minv = encode[:, :, pt_idxs[0], ff_idx].min()
        maxv = encode[:, :, pt_idxs[0], ff_idx].max()
        cb = ax.imshow((encode[:, :, pt_idxs[0], ff_idx] - minv) / (maxv - minv))
        ax.set_title(f'Normalized tx output {pt_idxs[0]}-{ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(3, 5, 7)
        minv = encode[:, :, pt_idxs[1], ff_idx].min()
        maxv = encode[:, :, pt_idxs[1], ff_idx].max()
        cb = ax.imshow((encode[:, :, pt_idxs[1], ff_idx] - minv) / (maxv - minv))
        ax.set_title(f'Normalized tx output {pt_idxs[1]}-{ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(3, 5, 8)
        minv = encode[:, :, pt_idxs[2], ff_idx].min()
        maxv = encode[:, :, pt_idxs[2], ff_idx].max()
        cb = ax.imshow((encode[:, :, pt_idxs[2], ff_idx] - minv) / (maxv - minv))
        ax.set_title(f'Normalized tx output {pt_idxs[2]}-{ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(3, 5, 9)
        minv = encode[:, :, pt_idxs[3], ff_idx].min()
        maxv = encode[:, :, pt_idxs[3], ff_idx].max()
        cb = ax.imshow((encode[:, :, pt_idxs[3], ff_idx] - minv) / (maxv - minv))
        ax.set_title(f'Normalized tx output {pt_idxs[3]}-{ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(3, 5, 10)
        minv = encode[:, :, pt_idxs[4], ff_idx].min()
        maxv = encode[:, :, pt_idxs[4], ff_idx].max()
        cb = ax.imshow((encode[:, :, pt_idxs[4], ff_idx] - minv) / (maxv - minv))
        ax.set_title(f'Normalized tx output {pt_idxs[4]}-{ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ff_idx = args.models.transformer.ff_dim // 3 * 2
        ax = fig.add_subplot(3, 5, 11)
        minv = encode[:, :, pt_idxs[0], ff_idx].min()
        maxv = encode[:, :, pt_idxs[0], ff_idx].max()
        cb = ax.imshow((encode[:, :, pt_idxs[0], ff_idx] - minv) / (maxv - minv))
        ax.set_title(f'Normalized tx output {pt_idxs[0]}-{ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(3, 5, 12)
        minv = encode[:, :, pt_idxs[1], ff_idx].min()
        maxv = encode[:, :, pt_idxs[1], ff_idx].max()
        cb = ax.imshow((encode[:, :, pt_idxs[1], ff_idx] - minv) / (maxv - minv))
        ax.set_title(f'Normalized tx output {pt_idxs[1]}-{ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(3, 5, 13)
        minv = encode[:, :, pt_idxs[2], ff_idx].min()
        maxv = encode[:, :, pt_idxs[2], ff_idx].max()
        cb = ax.imshow((encode[:, :, pt_idxs[2], ff_idx] - minv) / (maxv - minv))
        ax.set_title(f'Normalized tx output {pt_idxs[2]}-{ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(3, 5, 14)
        minv = encode[:, :, pt_idxs[3], ff_idx].min()
        maxv = encode[:, :, pt_idxs[3], ff_idx].max()
        cb = ax.imshow((encode[:, :, pt_idxs[3], ff_idx] - minv) / (maxv - minv))
        ax.set_title(f'Normalized tx output {pt_idxs[3]}-{ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        ax = fig.add_subplot(3, 5, 15)
        minv = encode[:, :, pt_idxs[4], ff_idx].min()
        maxv = encode[:, :, pt_idxs[4], ff_idx].max()
        cb = ax.imshow((encode[:, :, pt_idxs[4], ff_idx] - minv) / (maxv - minv))
        ax.set_title(f'Normalized tx output {pt_idxs[4]}-{ff_idx}\n' + 'Real min: %.3f, Real max: %.3f' % (minv, maxv))

        fig.suptitle("frame %d\n12: room back, 32: roof, 77: air vent, 85: board front, 87 shovel front" % (frame))
        
        canvas = fig.canvas
        buffer = io.BytesIO()  # 获取输入输出流对象
        canvas.print_png(buffer)  # 将画布上的内容打印到输入输出流对象
        data = buffer.getvalue()  # 获取流的值
        buffer.write(data)  # 将数据写入buffer 
        img = Image.open(buffer)  # 使用Image打开图片数据
        plots["txoutput"] = img
        plt.close()

    return plots


def test(model, device, dataset, save_name, args, resume_step):
    log_dir = os.path.join("experiments", args.index, "test-mic-z-02")
    # log_dir = os.path.join("experiments", args.index, "test-attn-temp{}".format(args.models.transformer.temp))
    os.makedirs(log_dir, exist_ok=True)

    # mic mvc3n-mic-1stattnM37-100pt-ff1-080808-coords.pt
    # mic = [53, 90, 74, 77, 45, 19, 40, 85, 37, 95, 32, 79, 27, 58, 50, 87, 61, 66, 71, 14, 24, 63,\
    #          11, 56, 29, 42, 22, 16, 69, 16, 48, 21, 43, 55, 6, 3, 84, 35, 34]
    # leg1 = [39, 83, 86, 60, 81, 78, 26]
    # leg2 = [18, 15, 52, 7, 73, 31]
    # leg3 = [94, 12, 70, 72, 46, 49]

    # # 100pt ff0.6
    shovel = [82, 87, 71, 53, 66, 61, 74, 58, 45, 79, 69, 98, 50, 37, 63, 40, 48]
    board_front = [92, 89, 85, 95, 68, 90, 96, 60, 94, 76, 81, 80, 93, 99, 39, 72, 97]
    board = [1, 2, 4, 5, 13, 9, 17, 30, 59, 72, 81, 90, 92, 89, 85, 95, 68, 96, 60, 94, 76, 80, 93, 99, 39, 97, 47, 34, 26, 21]
    cabin = [11, 19, 32, 24, 6, 12, 38, 43, 27, 16, 3]

    # # 5pt
    # shovel = [3]

    # # 200pt ff1.0
    # shovel = [163, 176, 160, 168, 150, 74, 108, 129, 155, 181, 171, 152, 95, "tbd"]

    points = model.points.detach().cpu().numpy()
    for idx in shovel:
        print(idx, points[idx])
        points[idx, 2] += -0.2
        print(idx, points[idx])
    model.points = torch.nn.Parameter(torch.from_numpy(points).to(model.points.device), requires_grad=True)

    # points = model.points.detach().cpu().numpy()
    # for idx in shovel:
    #     print(idx, points[idx])
    #     points[idx, 2] += -0.3
    #     # points[idx, 0] += -0.5
    #     print(idx, points[idx])

    # for i in range(points.shape[0]):
    #     if points[i, 1] < -0.15 and points[i, 2] > 0.5: # shovel
    #         points[i, 2] += -0.2
    #         print(i, points[i])

    testloader = get_testloader(dataset, args.dataset)
    print("testloader:", testloader)

    loss_fn = get_loss(args.training.losses)

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    test_losses = []
    test_psnrs = []

    frames = {}
    for frame, batch in enumerate(testloader):
        plots = test_step(frame, len(testloader), model, device, dataset, batch, loss_fn, args, test_losses, test_psnrs)

        if plots:
            for key, value in plots.items():
                if key not in frames:
                    frames[key] = []
                frames[key].append(value)
    
    if frames:
        for key, value in frames.items():
            # f = os.path.join(log_dir, f"{key}.mp4")
            f = os.path.join(log_dir, f"{args.index}-{key}-{save_name}-step{resume_step}.mp4")
            imageio.mimwrite(f, value, fps=30, quality=10)

    test_loss = np.mean(test_losses)
    test_psnr = np.mean(test_psnrs)

    print(f"Avg test loss: {test_loss:.4f}, test PSNR: {test_psnr:.4f}")


def main(args, save_name):
    model = get_model(args)
    dataset = get_testdataset(args.dataset)

    resume_step = 0
    if args.test.load_path:
        model.load_state_dict(torch.load(os.path.join("experiments", args.test.load_path, "model.pth")))
    else:
        try:
            resume_step = 250000
            model.load_state_dict(torch.load(os.path.join("experiments", args.index, f"model_{resume_step}.pth")))
        except:
            model_state_dict = torch.load(os.path.join("experiments", args.index, "model.pth"))
            for step, state_dict in model_state_dict.items():
                resume_step = step
                model.load_state_dict(state_dict)
    print("resume_step:", resume_step)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(model, device, dataset, save_name, args, resume_step)


if __name__ == '__main__':

    args = parse_args()
    with open(args.opt, 'r') as f:
        config = yaml.safe_load(f)

    log_dir = os.path.join("experiments", config['index'])
    os.makedirs(log_dir, exist_ok=True)

    shutil.copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))
    shutil.copyfile(args.opt, os.path.join(log_dir, os.path.basename(args.opt)))

    setup_seed(config['seed'])

    # print(config["dataset"], type(config["dataset"]))

    for dataset in config['test']['datasets']:
        name = dataset['name']
        print(name, dataset)
        config['dataset'].update(dataset)
        args = DictAsMember(config)
        main(args, name)

    # print(config["dataset"], type(config["dataset"]))
    # config["dataset"].update(config["test"]["dataset"])
    # print(config["dataset"], type(config["dataset"]))
    # config = DictAsMember(config)
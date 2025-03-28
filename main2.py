import argparse

from engin import Engin


def main(args):
    engin = Engin(
        model_path=args.model_path,
        output_dir=args.output_dir,
        extension=args.extension,
        manual_select=args.manual,
        scale=args.scale
    )
    engin.run(args.input_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="图片文件夹")
    parser.add_argument("--output_dir", type=str, required=True, help="截图后的图片保存文件夹")
    parser.add_argument("--model_path", type=str, default="./pytorch_model.pt")
    parser.add_argument("--scale", type=float, default=0.5, help="缩放比例，小于1时缩小图片")
    parser.add_argument("--manual", action="store_true", help="是否手动选择四个点")
    parser.add_argument("--extension", nargs="+", default=["jpg", "jpeg", "png"], help="图片扩展名")

    args = parser.parse_args()
    main(args)

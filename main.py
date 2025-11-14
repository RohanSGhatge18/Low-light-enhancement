# main.py
import os
import argparse
import cv2
from enhance import enhance_image
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Low-light image enhancement pipeline.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', help='Path to input image')
    group.add_argument('--input_dir', help='Path to folder containing images to process (jpg/png etc.)')

    p.add_argument('--output', help='Path to output image (when --input used)')
    p.add_argument('--out_dir', help='Output directory when --input_dir used')
    p.add_argument('--show', action='store_true', help='Show before/after window (press any key to continue)')
    p.add_argument('--denoise', action='store_true', help='Apply denoising (default: on when not specified? use flag to enable)')
    p.add_argument('--h', type=float, default=10.0, help='h for color NLMeans (strength)')
    p.add_argument('--hColor', type=float, default=10.0, help='hColor for color NLMeans')
    p.add_argument('--clahe_clip', type=float, default=2.0, help='CLAHE clip limit')
    p.add_argument('--clahe_tile', type=int, default=8, help='CLAHE tile grid size (int)')
    p.add_argument('--gamma', type=float, default=1.1, help='Gamma correction (>1 brightens)')
    p.add_argument('--sharpen', type=float, default=1.0, help='Sharpen strength (0 = off, 1 = default)')
    p.add_argument('--use_skimage', action='store_true', help='Use scikit-image nlmeans for grayscale (optional)')
    return p.parse_args()

def process_file(path, out_path, args):
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Cannot read {path}")
        return
    result = enhance_image(img_bgr=img,
                           denoise=args.denoise,
                           h=args.h,
                           hColor=args.hColor,
                           clahe_clip=args.clahe_clip,
                           clahe_tile=args.clahe_tile,
                           gamma=args.gamma,
                           sharpen_strength=args.sharpen,
                           use_skimage_nlmeans=args.use_skimage)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    cv2.imwrite(out_path, result)
    print(f"[SAVED] {out_path}")

    if args.show:
        # stack before/after horizontally for quick comparison
        before = cv2.resize(img, (0,0), fx=0.6, fy=0.6)
        after = cv2.resize(result, (before.shape[1], before.shape[0]))
        stacked = np.hstack([before, after])
        cv2.imshow('Before | After', stacked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    args = parse_args()
    if args.input:
        out = args.output or os.path.splitext(args.input)[0] + '_enhanced.jpg'
        process_file(args.input, out, args)
    else:
        # folder mode
        indir = args.input_dir
        outdir = args.out_dir or os.path.join(indir, 'enhanced')
        os.makedirs(outdir, exist_ok=True)
        supported = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        for fname in os.listdir(indir):
            if fname.lower().endswith(supported):
                inp = os.path.join(indir, fname)
                outp = os.path.join(outdir, os.path.splitext(fname)[0] + '_enhanced.jpg')
                process_file(inp, outp, args)

if __name__ == '__main__':
    main()

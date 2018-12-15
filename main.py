import argparse, os, torch
from GAN import GAN
from CGAN import CGAN
from LSGAN import LSGAN
from DRAGAN import DRAGAN
from ACGAN import ACGAN
from WGAN import WGAN
from WGAN_GP import WGAN_GP
from infoGAN import infoGAN
from EBGAN import EBGAN
from BEGAN import BEGAN

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed'],
                        help='The name of dataset')
    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=2, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=False)#True
    parser.add_argument('--benchmark_mode', type=bool, default=True)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments

    args = parse_args()
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True
        '''
        如果网络的输入数据维度或类型上变化不大，设置
        torch.backends.cudnn.benchmark = true
        可以增加运行效率；如果网络的输入数据在每次
        iteration都变化的话，会导致cudnn
        每次都会去寻找一遍最优配置，这样反而会降低运行效率。
        '''
        # declare instance for GAN
    if args.gan_type == 'GAN':
        print("GAN is "+args.gan_type)
        gan = GAN(args)
    elif args.gan_type == 'CGAN':
        print("GAN is " + args.gan_type)
        gan = CGAN(args)
    elif args.gan_type == 'ACGAN':
        print("GAN is " + args.gan_type)
        gan = ACGAN(args)
    elif args.gan_type == 'infoGAN':
        print("GAN is " + args.gan_type)
        gan = infoGAN(args, SUPERVISED=False)
    elif args.gan_type == 'EBGAN':
        print("GAN is " + args.gan_type)
        gan = EBGAN(args)
    elif args.gan_type == 'WGAN':
        print("GAN is " + args.gan_type)
        gan = WGAN(args)
    elif args.gan_type == 'WGAN_GP':
        print("GAN is " + args.gan_type)
        gan = WGAN_GP(args)
    elif args.gan_type == 'DRAGAN':
        print("GAN is " + args.gan_type)
        gan = DRAGAN(args)
    elif args.gan_type == 'LSGAN':
        print("GAN is " + args.gan_type)
        gan = LSGAN(args)
    elif args.gan_type == 'BEGAN':
        print("GAN is " + args.gan_type)
        gan = BEGAN(args)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)

        # launch the graph in a session
    gan.train()
    print(" [*] Training finished!")

    # visualize learned generator
    gan.visualize_results(args.epoch)
    print(" [*] Testing finished!")

if __name__ == '__main__':
    main()

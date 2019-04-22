import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = 1  # no visdom display
    epochs = opt.which_epoch.split(',')
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    for epoch in epochs:
        opt.which_epoch = epoch

        model = create_model(opt)
        model.setup(opt)
        # create website
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
        # test
        count = 0
        for i, data in enumerate(dataset):
            count += 1
            if i >= opt.how_many:
                break
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            if i % 5 == 0:
                print('processing (%04d)-th image... %s' % (i, img_path))
            print('count ', count)
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        webpage.save()
    # python test.py --dataroot data/datasets/ken/etl_952_singlechar_size_64/ --name  1252_dense5 --gpu_ids 0  --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG densenet_5blocks --no_dropout
    # --which_epoch 200 --how_many 100 --input_nc 1 --output_nc 1
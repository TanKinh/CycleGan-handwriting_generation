import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse() # Trainning option
    data_loader = CreateDataLoader(opt) # Create a dataset given opt.dataset_mode and other options
    dataset = data_loader.load_data() # Get dataset
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt) # create a model given opt.model and other options
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data) # unpack data from dataset and apply preprocessing
            model.optimize_parameters() # calculate loss functions, get gradients, update network weights

            if total_steps % opt.display_freq == 0: # visualize on visdom
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0: # print trainning loss and log to disk
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0: # save latest model to disk
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0: #save model to disk
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
'''
python data/prepare_data/prepare_casia.py --source data/sources/1252-c.gnt --font data/fonts/simhei.ttf --fontSize 116 --offset 6 --percent 100 --ratioA 0.1 --ratioB 0.1
python train.py --dataroot data/datasets/1252-c_116_6_0.1/ --name 1252_dense5_0.1_64 --gpu_ids 0 --which_direction BtoA --loadSize 64 --fineSize 64 --which_model_netG densenet_5blocks --no_dropout --input_nc 1 --output_nc 1
python test_line.py --name 1252_dense5_0.1_64 --font data/fonts/simhei.ttf --font_size 116 --offset 6 --which_model_netG densenet_5blocks --no_dropout --gpu_ids 0 --input_nc 1 --output_nc 1 --which_direction BtoA --loadSize 64 --fineSize 64 --which_epoch 200
python test.py --dataroot data/datasets/1252-c_116_6_0.1/ --name  1252_dense5_0.1_64 --gpu_ids 0  --which_direction BtoA --loadSize 64 --fineSize 64 --which_model_netG densenet_5blocks --no_dropout --which_epoch 200 --how_many 100 --input_nc 1 --output_nc 1

'''
'''
python train.py --dataroot data/datasets/1252-c_116_6_0.1/ --name 1252_dense5_0.1_64_A_B --gpu_ids 0 --which_direction AtoB --loadSize 64 --fineSize 64 --which_model_netG densenet_5blocks --no_dropout --inpu
t_nc 1 --output_nc 1
python test_line_hand.py --name 1252_dense5_0.1_64_A_B --font data/fonts/simhei.ttf --font_size 116 --offset 6 --which_model_netG densenet_5blocks --no_dropout --gpu_ids 0 --input_nc 1 --output_nc 1 --which_direction AtoB --loadSize 128 --fineSize 128 --which_epoch 200
七
'''

'''
# Restnet-6 0.1 128
python data/prepare_data/prepare_casia.py --source data/sources/1252-c.gnt --font data/fonts/simhei.ttf --fontSize 116 --offset 6 --percent 100 --ratioA 0.1 --ratioB 0.1
python train.py --dataroot data/datasets/1252-c_116_6_0.1/ --name 1252_resnet6 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG resnet_6blocks --no_dropout --input_nc 1 --output_nc 1
python test.py --dataroot data/datasets/1252-c_116_6_0.1/ --name 1252_resnet6 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG resnet_6blocks --no_dropout --which_epoch 200 --how_many 100 --input_nc 1 --output_nc 1
python test_line_hand.py --name 1252_resnet6 --font data/fonts/simhei.ttf --font_size 116 --offset 6 --which_model_netG resnet_6blocks --no_dropout --gpu_ids 0 --input_nc 1 --output_nc 1 --which_direction BtoA --loadSize 128 --fineSize 128 --which_epoch 200

64 4.5hour
python train.py --dataroot data/datasets/1252-c_116_6_0.1/ --name 1252_resnet6_0.1_64 --gpu_ids 0 --which_direction BtoA --loadSize 64 --fineSize 64 --which_model_netG resnet_6blocks --no_dropout --input_nc 1 --output_nc 1
python test.py --dataroot data/datasets/1252-c_116_6_0.1/ --name 1252_resnet6_0.1_64 --gpu_ids 0 --which_direction BtoA --loadSize 64 --fineSize 64 --which_model_netG resnet_6blocks --no_dropout --which_epoch 200,168,180 --how_many 100 --input_nc 1 --output_nc 1
python test_line_hand.py --name 1252_resnet6_0.1_64 --font data/fonts/simhei.ttf --font_size 116 --offset 6 --which_model_netG resnet_6blocks --no_dropout --gpu_ids 0 --input_nc 1 --output_nc 1 --which_direction BtoA --loadSize 128 --fineSize 128 --which_epoch 200
'''

''' Densenet-6
python train.py --dataroot data/datasets/1252-c_116_6_0.1/ --name 1252_densenet6 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG densenet_6blocks --no_dropout --input_nc 1 --output_nc 1
python test.py --dataroot data/datasets/1252-c_116_6_0.1/ --name 1252_densenet6 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG densenet_6blocks --no_dropout --which_epoch 200 --how_many 100 --input_nc 1 --output_nc 1
python test_line_hand.py --name 1252_densenet6 --font data/fonts/simhei.ttf --font_size 116 --offset 6 --which_model_netG densenet_5blocks --no_dropout --gpu_ids 0 --input_nc 1 --output_nc 1 --which_direction BtoA --loadSize 128 --fineSize 128 --which_epoch 200
'''

''' unet_128
python train.py --dataroot data/datasets/1252-c_116_6_0.1/ --name 1252_unet_128 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG unet_128 --no_dropout --input_nc 1 --output_nc 1
python test.py --dataroot data/datasets/1252-c_116_6_0.1/ --name 1252_unet_128 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG unet_128 --no_dropout --which_epoch 165,180,190,200 --how_many 3379 --input_nc 1 --output_nc 1
'''

''' ration 0.2
python train.py --dataroot data/datasets/1252-c_116_6_0.2/ --name 1252_resnet6_0.2 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG resnet_6blocks --no_dropout --input_nc 1 --output_nc 1
python test.py --dataroot data/datasets/1252-c_116_6_0.2/ --name 1252_resnet6_0.2 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG resnet_6blocks --no_dropout --which_epoch 200 --how_many 3003 --input_nc 1 --output_nc 1
'''


"""Densenet5 
python train.py --dataroot data/datasets/1252-c_116_6_0.1/ --name 1252_densenet5_0.1_128 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG densenet_5blocks --no_dropout --input_nc 1 --output_nc 1
python test.py --dataroot data/datasets/1252-c_116_6_0.1/ --name 1252_densenet5_0.1_128 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG densenet_5blocks --no_dropout --which_epoch 200,180,190,165 --how_many 100 --input_nc 1 --output_nc 1
"""
###### ratio 0.9, densenet_5blocks
# python test.py --dataroot data/datasets/1252-c_116_6_0.1/ --name 1252_dense5_1.0_full_data --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG densenet_5blocks --no_dropout --which_epoch 200 --how_many 3379 --input_nc 1 --output_nc 1

#---------------ETL - etl_952_singlechar_size_64 -----------------#
''' Resnet_6
python data/prepare_data/prepare_ETL8.py
python train.py --dataroot data/datasets/etl_952_singlechar_size_64/ --name ETL_resnet6_0.1_128 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG resnet_6blocks --no_dropout --input_nc 1 --output_nc 1
python test.py --dataroot data/datasets/etl_952_singlechar_size_64/ --name ETL_resnet6_0.1_128 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG resnet_6blocks --no_dropout --which_epoch 165,180,190,200 --how_many 856 --input_nc 1 --output_nc 1
python test_line_hand.py --name ETL_resnet6_0.1_128 --font data/fonts/simhei.ttf --font_size 116 --offset 6 --which_model_netG resnet_6blocks --no_dropout --gpu_ids 0 --input_nc 1 --output_nc 1 --which_direction BtoA --loadSize 64 --fineSize 64 --which_epoch 200

--64--
python train.py --dataroot data/datasets/etl_952_singlechar_size_64/ --name ETL_resnet6_0.1_64 --gpu_ids 0 --which_direction BtoA --loadSize 64 --fineSize 64 --which_model_netG resnet_6blocks --no_dropout --input_nc 1 --output_nc 1
python test.py --dataroot data/datasets/etl_952_singlechar_size_64/ --name ETL_resnet6_0.1_64 --gpu_ids 0 --which_direction BtoA --loadSize 64 --fineSize 64 --which_model_netG resnet_6blocks --no_dropout --which_epoch 165,180,190,200 --how_many 856 --input_nc 1 --output_nc 1
python test_line_hand.py --name ETL_resnet6_0.1_64 --font data/fonts/simhei.ttf --font_size 116 --offset 6 --which_model_netG resnet_6blocks --no_dropout --gpu_ids 0 --input_nc 1 --output_nc 1 --which_direction BtoA --loadSize 64 --fineSize 64 --which_epoch 200

'''

''' Densenet-5
python train.py --dataroot data/datasets/etl_952_singlechar_size_64/ --name ETL_densenet5_0.1_128 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG densenet_5blocks --no_dropout --input_nc 1 --output_nc 1
python test.py --dataroot data/datasets/etl_952_singlechar_size_64/ --name ETL_densenet5_0.1_128 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG densenet_5blocks --no_dropout --which_epoch 165,180,190,200 --how_many 856 --input_nc 1 --output_nc 1
'''

''' Unet128
python train.py --dataroot data/datasets/etl_952_singlechar_size_64/ --name ETL_unet_128_0.1_128 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG unet_128 --no_dropout --input_nc 1 --output_nc 1
python test.py --dataroot data/datasets/etl_952_singlechar_size_64/ --name ETL_unet_128_0.1_128 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG unet_128 --no_dropout --which_epoch 165,180,190,200 --how_many 856 --input_nc 1 --output_nc 1
'''
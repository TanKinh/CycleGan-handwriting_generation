import os
import torch

from options.test_line_options import TestLineOptions
from data import CreateDataLoader
from data.base_dataset import get_transform
from models import create_model
from util.visualizer import save_image
from util import html
from PIL import Image, ImageFont, ImageDraw
from functools import reduce
import json

def draw_single_char(ch, font, canvas_size=128, x_offset=26, y_offset=36):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img.convert('RGB')


def gen_line(opt):
    result_img_names = []

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    font = ImageFont.truetype(opt.font, size=opt.font_size)

    epochs = opt.which_epoch.split(',')
    count = 0
    for epoch in epochs:
        opt.which_epoch = epoch
        model = create_model(opt)
        model.setup(opt)

        with open('labels.json', encoding='utf-8') as f:
            json_load = json.load(f)
            for line in json_load:
                results = []
                inputs = []
                text = json_load[line]
                count += 1
                if (count > 265846):
                    print(count, '  ',line)
                    for ch in text:
                        img = draw_single_char(ch, font, x_offset=opt.offset, y_offset=opt.offset)

                        transform = get_transform(opt)

                        img = transform(img)

                        if opt.input_nc == 1:  # RGB to gray
                            tmp = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
                            img = tmp.unsqueeze(0)


                        img = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])
                        inputs.append(img)

                        model.set_input_real_A(img)
                        model.test_fake_B()
                        results.append(model.fake_B)

                    result = reduce((lambda x, y: torch.cat((x, y), -1)), results)
                    input_img = reduce((lambda x, y: torch.cat((x, y), -1)), inputs)

                    result_img_name = line
                    # result_img_name = 'result_' + opt.name + '_' + str(epoch) + '_' + text +  '.png'
                    input_img_name = 'input_' + opt.name + '_' + text + '.png'
                    result_img_names.append(result_img_name)

                    save_image(opt.results_dir, result, result_img_name, aspect_ratio=opt.aspect_ratio)
                    print(opt.results_dir)
                    #save_image(opt.results_dir, input_img, input_img_name, aspect_ratio=opt.aspect_ratio)

    return {'input': input_img_name, 'result': result_img_names}

def gen_line_txt(forder, text, name, opt):
    result_img_names = []

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    font = ImageFont.truetype(opt.font, size=opt.font_size)

    epochs = opt.which_epoch.split(',')
    for epoch in epochs:
        results = []
        inputs = []
        opt.which_epoch = epoch
        model = create_model(opt)
        model.setup(opt)

        for ch in text:
            img = draw_single_char(ch, font, x_offset=opt.offset, y_offset=opt.offset)

            transform = get_transform(opt)

            img = transform(img)

            if opt.input_nc == 1:  # RGB to gray
                tmp = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
                img = tmp.unsqueeze(0)


            img = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])
            inputs.append(img)

            model.set_input_real_A(img)
            model.test_fake_B()
            results.append(model.fake_B)

        result = reduce((lambda x, y: torch.cat((x, y), -1)), results)
        input_img = reduce((lambda x, y: torch.cat((x, y), -1)), inputs)

        result_img_name = name
        input_img_name = 'input_' + opt.name + '_' + text + '.png'
        result_img_names.append(result_img_name)

        save_image(forder, result, result_img_name, aspect_ratio=opt.aspect_ratio)
        #save_image(opt.results_dir, input_img, input_img_name, aspect_ratio=opt.aspect_ratio)

    return {'input': input_img_name, 'result': result_img_names}


if __name__ == '__main__':
    # text = input("Input text: ")
    # print(text)
    filepath = 'Form123_bank_branch_ocr_Val_label_new100Fuku.txt'  
    images = list(open(filepath, encoding='utf-8'))
    opt = TestLineOptions().parse()
    count = 0
    for image in images:
        label = image.split('|')[1]
        label = label.rstrip()
        name = image.split('/')[1]
        forder = opt.results_dir + str('/') + image.split('/')[0] 
        if not os.path.exists(forder):
            os.makedirs(forder)
        name = name.split('|')[0]
        print(forder,'   ',label,'   ', name)
        gen_line_txt(forder, label, name, opt)
'''
    count = 0
    opt = TestLineOptions().parse()
    # with open('labels.json', encoding='utf-8') as f:
    #     json_load = json.load(f)
    #     for line in json_load:
    #         count += 1
    #         if (count > 5700):
    gen_line(opt)
            # print(count)
            # if count == 10:
            #     break
'''         
# python test_line.py --name ETL_densenet5_0.1_128 --font data/fonts/simhei.ttf --font_size 116 --offset 6 --which_model_netG densenet_5blocks --no_dropout --gpu_ids -2 --input_nc 1 --output_nc 1 --which_direction BtoA --loadSize 128 --fineSize 128 --which_epoch 200
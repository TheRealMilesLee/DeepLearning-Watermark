#!/bin/sh

# echo指令测试
echo "start sh_test.sh"

#python  pix2pix_GrayVer_dct2_withoutProcess.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_dct/train" \
#                                   --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/train_newDCT_noTanh_255(epoch=6)" \
#                                   --mode="train"


python  pix2pix_GrayVer_dct2_withoutProcess.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_dct/test" \
                                   --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/test_newDCT_noTanh_255(epoch=6)" \
                                   --checkpoint="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/train_newDCT_noTanh_255(epoch=6)" \
                                   --mode="test"


python ../dct_watermark/DCT-de_forTrain.py \
--source_path="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/test_newDCT_noTanh_255(epoch=6)/images" \
--output_path="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/extract_newDCT_noTanh_255(epoch=6)"

#python pix2pix.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_lsbm/test" \
#                  --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsbm/test_pix2pix" \
#                  --checkpoint="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsbm/train_pix2pix" \
#                  --mode="test"
#
#
#python pix2pix_pixel-expanded.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_lsbm/test" \
#                                 --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsbm/test_pixel-expanded" \
#                                 --checkpoint="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsbm/train_pixel-expanded" \
#                                 --mode="test"
#
#python pix2pix.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_lsbmr/test" \
#                  --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsbmr/test_pix2pix" \
#                  --checkpoint="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsbmr/train_pix2pix" \
#                  --mode="test"
#
#
#python pix2pix_pixel-expanded.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_lsbmr/test" \
#                                 --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsbmr/test_pixel-expanded" \
#                                 --checkpoint="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsbmr/train_pixel-expanded" \
#                                 --mode="test"
#
#
#python pix2pix.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_lsb/test" \
#                  --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsb/test_pix2pix" \
#                  --checkpoint="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsb/train_pix2pix" \
#                  --mode="test"
#
#python ../lsb_watermark/lsb-de-cv-train.py \
#        --input_path="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsb/test_pix2pix/images" \
#        --output_path="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsb/extract_pix2pix"
#
#python pix2pix_pixel-expanded.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_lsb/test" \
#                                 --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsb/test_pixel-expanded" \
#                                 --checkpoint="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsb/train_pixel-expanded" \
#                                 --mode="test"
#
#python ../lsb_watermark/lsb-de-cv-train.py \
#        --input_path="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsb/test_pixel-expanded/images" \
#        --output_path="/home/wangruowei/PycharmProjects/watermark2020/experment_results/lsb/extract_pixel-expanded"

#python  pix2pix_GrayVer_dct2_bn.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_dct/train" \
#                                   --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/train_newDCT_noTanh_bn(l1_weight=100)" \
#                                   --mode="train" \
#                                   --l1_weight=100.0
#
#python  pix2pix_GrayVer_dct2_bn.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_dct/test" \
#                                   --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/test_newDCT_noTanh_bn(l1_weight=100)" \
#                                   --checkpoint="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/train_newDCT_noTanh_bn(l1_weight=100)" \
#                                   --mode="test"\
#                                   --l1_weight=100.0
#
#python ../dct_watermark/DCT-de_forTrain.py \
#--source_path="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/test_newDCT_noTanh_bn(l1_weight=100)/images" \
#--output_path="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/extract_newDCT_noTanh_bn(l1_weight=100)"
#
## -----分割线-----
#
#python  pix2pix_GrayVer_dct2_bn.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_dct/train" \
#                                   --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/train_newDCT_noTanh_bn(l1_weight=200)" \
#                                   --mode="train" \
#                                   --l1_weight=200.0
#
#python  pix2pix_GrayVer_dct2_bn.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_dct/test" \
#                                   --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/test_newDCT_noTanh_bn(l1_weight=200)" \
#                                   --checkpoint="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/train_newDCT_noTanh_bn(l1_weight=200)" \
#                                   --mode="test" \
#                                   --l1_weight=200.0
#
#python ../dct_watermark/DCT-de_forTrain.py \
#--source_path="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/test_newDCT_noTanh_bn(l1_weight=200)/images" \
#--output_path="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/extract_newDCT_noTanh_bn(l1_weight=200)"
#
## -----分割线-----
#
#python  pix2pix_GrayVer_dct2_bn.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_dct/train" \
#                                   --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/train_newDCT_noTanh_bn(l1_weight=300)" \
#                                   --mode="train" \
#                                   --l1_weight=300.0
#
#python  pix2pix_GrayVer_dct2_bn.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_dct/test" \
#                                   --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/test_newDCT_noTanh_bn(l1_weight=300)" \
#                                   --checkpoint="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/train_newDCT_noTanh_bn(l1_weight=300)" \
#                                   --mode="test" \
#                                   --l1_weight=300.0
#
#python ../dct_watermark/DCT-de_forTrain.py \
#--source_path="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/test_newDCT_noTanh_bn(l1_weight=300)/images" \
#--output_path="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/extract_newDCT_noTanh_bn(l1_weight=300)"
#
## -----分割线-----
#
#python  pix2pix_GrayVer_dct2_bn.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_dct/train" \
#                                   --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/train_newDCT_noTanh_bn(l1_weight=50)" \
#                                   --mode="train" \
#                                   --l1_weight=50.0
#
#python  pix2pix_GrayVer_dct2_bn.py --input_dir="/home/wangruowei/PycharmProjects/watermark2020/data/pair_dct/test" \
#                                   --output_dir="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/test_newDCT_noTanh_bn(l1_weight=50)" \
#                                   --checkpoint="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/train_newDCT_noTanh_bn(l1_weight=50)" \
#                                   --mode="test" \
#                                   --l1_weight=50.0
#
#python ../dct_watermark/DCT-de_forTrain.py \
#--source_path="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/test_newDCT_noTanh_bn(l1_weight=50)/images" \
#--output_path="/home/wangruowei/PycharmProjects/watermark2020/experment_results/dct/extract_newDCT_noTanh_bn(l1_weight=50)"

echo "finished"
import random
from Utils import *
from models import Network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr as SROCC


def test_model():

    load_all_model_path = './PreTrainedModels/'
    valset_dir = './Datasets/HUST_LFI_5x32x64/'
    dataset_name = 'HUST_LFI'
    scene_list = ['I01C00', 'I02C00', 'I03C00', 'I04C00', 'I05C00',
                  'I06C00', 'I07C00', 'I08C00', 'I09C00', 'I10C00',
                  'I11C00', 'I12C00', 'I13C00', 'I14C00', 'I15C00',
                  'I16C00', 'I17C00', 'I18C00', 'I19C00', 'I20C00']

    test_scene_num = 4
    distorted_num = 5
    scene_num = 20

    Dist_DICT = {
        'JPEG2000': ['01', '02', '03', '04', '05'],
        'HEVC': ['06', '07', '08', '09', '10'],
        'VVC': ['11', '12', '13', '14', '15'],
        'Cheng': ['16', '17', '18', '19', '20'],
        'ELiC': ['21', '22', '23', '24', '25'],
        'DCVC_FM': ['26', '27', '28', '29', '30'],
        'DCVC_RT': ['31', '32', '33', '34', '35'],
        'JPEGPleno': ['36', '37', '38', '39', '40'],
        'SADN': ['41', '42', '43', '44', '45'],
        'LFIC_DRASC': ['46', '47', '48', '49', '50']
    }

    device = 'cuda:0'
    net = Network().to(device)
    cudnn.benchmark = True

    test_scene_id_list = []
    for i_scene in range(0, scene_num, 4):
        test_scene_id_list.append(list(range(i_scene, i_scene + 4)))

    # =======================
    # 遍历所有失真类型
    # =======================
    for dist in Dist_DICT.keys():

        print("\n==============================")
        print("Testing distortion:", dist)
        print("==============================")

        all_model = os.listdir(load_all_model_path)

        label_list = np.zeros([test_scene_num * distorted_num, len(all_model)])
        data_list = np.zeros([test_scene_num * distorted_num, len(all_model)])
        val_SRCC_all = []

        for id, model_name in enumerate(test_scene_id_list):

            load_model_path = load_all_model_path + '/' + \
                              str(model_name[0]) + '_' + str(model_name[-1]) + \
                              '/MMBLFQA_epoch70.pth.tar'

            model = torch.load(load_model_path, map_location={'cuda:0': device})
            net.load_state_dict(model['state_dict'])
            net.eval()

            index = 0

            for test_scene in model_name:

                image_path = valset_dir + '/' + scene_list[test_scene]
                image_list = os.listdir(image_path)

                for test_image in image_list:

                    if dataset_name == 'HUST_LFI':
                        if str(test_image.split('C')[1]) not in Dist_DICT[dist]:
                            continue
                    else:
                        print("=> no found dataset name")

                    patch_path = image_path + '/' + test_image
                    patch_list = os.listdir(patch_path)

                    output_list = 0

                    for val_patch in patch_list:
                        each_patch_path = patch_path + '/' + val_patch

                        with h5py.File(each_patch_path, 'r') as hf:
                            label = np.array(hf.get('score_label'))
                            data = np.array(hf.get('data'))

                            data = data / 255
                            data = np.expand_dims(data, axis=0)
                            data = np.expand_dims(data, axis=0)
                            data = torch.from_numpy(data.copy())
                            data = Variable(data).to(device)

                        with torch.no_grad():
                            out_score, _, _ = net(data)

                        output_list += out_score.cpu().numpy()

                    label_list[index, id] = label.item()
                    data_list[index, id] = output_list.item() / len(patch_list)
                    index += 1

            val_SRCC = SROCC(data_list[:, id], label_list[:, id]).correlation
            val_SRCC_all.append(val_SRCC)

            print("Test scenes:", model_name)
            print('SROCC :----    %f' % val_SRCC)

        print('Average SROCC (%s): ----   %f' % (dist, np.mean(val_SRCC_all)))

        # =======================
        # 保存每种失真的结果
        # =======================
        save_path = './Results/MMBLFQA_result_' + dataset_name + '_' + dist + '.h5'
        f = h5py.File(save_path, 'w')
        f.create_dataset('predict_data', data=data_list)
        f.create_dataset('score_label', data=label_list)
        f.close()

        print("Saved to:", save_path)

if __name__ == '__main__':
    # set random seed
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    test_model()

import torch
from Utils import *
from scipy.stats import spearmanr as SRCC


def val(net, valset_dir, test_scene_id, device='cuda:0', type='val'):
    label_list = []
    data_list = []
    criterion = torch.nn.MSELoss()

    scene_list = [
        'I01C00', 'I02C00', 'I03C00', 'I04C00', 'I05C00',
        'I06C00', 'I07C00', 'I08C00', 'I09C00', 'I10C00',
        'I11C00', 'I12C00', 'I13C00', 'I14C00', 'I15C00',
        'I16C00', 'I17C00', 'I18C00', 'I19C00', 'I20C00'
    ]

    with torch.no_grad():

        for test_scene in test_scene_id:

            image_path = os.path.join(valset_dir, scene_list[test_scene])

            for test_image in os.listdir(image_path):

                patch_path = os.path.join(image_path, test_image)
                patch_list = os.listdir(patch_path)

                output_list = 0

                for val_patch in patch_list:
                    each_patch_path = patch_path + '/' + val_patch

                    with h5py.File(each_patch_path, 'r') as hf:
                        label = float(np.array(hf['score_label']).squeeze())

                        data = np.array(hf['data']) / 255.0
                        data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().to(device)

                    with torch.no_grad():
                        out_score, _, _ = net(data)

                    output_list += out_score.item()

                label_list.append(label)
                data_list.append(output_list / len(patch_list))

    loss = criterion(
        torch.tensor(data_list),
        torch.tensor(label_list)
    )

    val_SRCC = SRCC(data_list, label_list).correlation
    val_SRCC = 0 if np.isnan(val_SRCC) else val_SRCC

    print(type + ' SRCC :----    %f' % val_SRCC)
    print(type + ' LOSS :----    %f' % loss)

    return val_SRCC

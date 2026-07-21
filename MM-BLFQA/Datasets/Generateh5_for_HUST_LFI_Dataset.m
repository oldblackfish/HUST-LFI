clc;close all;clear;

dataset_path = 'F:\HUST-LFI\Distorted\'; % Set the dataset path here
savepath = '.\HUST_LFI_5x32x64\'; % Set the save path here

% for HUST_LFI dataset
% =========================
T = readtable('HUST_LFI_MOS_Results.csv');

Names = T.Distortion;
Mos  = T.MOS;

Distorted_sceneNum = 1000; 

angRes = 5;
patchsize = 32;
stride = 64;

inum = 1;
for iScene = 1 : Distorted_sceneNum

    cls_name = extractAfter(Names{iScene}, 'C');
    cls_id   = str2double(cls_name);   % '01' → 1

    % 安全检查
    if isnan(cls_id) || cls_id < 1 || cls_id > 50
        error('未匹配的值: %s', cls_name);
    end

    % ========= 失真类别 (0~9) =========
    cls = floor((cls_id - 1) / 5);

    % ========= 失真等级 (0~4) =========
    level = mod(cls_id - 1, 5);
    
    tic;
    idx = 1;
    h5_savedir = [savepath, '\', [Names{iScene}(1:end-2), '00'], '\', Names{iScene}];
    if exist(h5_savedir, 'dir')==0
        mkdir(h5_savedir);
    end
    dataPath = [dataset_path, Names{iScene}, '.png'];
    LF = imread(dataPath);
    [H_raw, W_raw, ~] = size(LF);
    LF = permute(reshape(LF,[9, H_raw/9, 9, W_raw/9, 3]),[1,3,2,4,5]);
    [U, V, H_sai, W_sai, ~] = size(LF);

    LF = LF(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), :, :, :);
    [U, V, H, W, ~] = size(LF);
    
    dis_mli = single(zeros(U * patchsize, V * patchsize));

    label = Mos(iScene);
    for h = 1 : stride : H - patchsize + 1
        for w = 1 : stride : W - patchsize + 1
            for u = 1 : U
                for v = 1 : V
                    temp_dis = squeeze(LF(u, v, h : h+patchsize-1, w : w+patchsize-1, :));
                    temp_dis = rgb2ycbcr(temp_dis);
                    temp_dis = squeeze(temp_dis(:,:,1));
                    dis_mli(u:angRes:U * patchsize, v:angRes:V * patchsize) = temp_dis;
                end
            end
            SavePath_H5_name = [h5_savedir, '/', num2str(idx,'%06d'),'.h5'];
            h5create(SavePath_H5_name, '/data', size(dis_mli), 'Datatype', 'single');
            h5write(SavePath_H5_name, '/data', single(dis_mli), [1,1], size(dis_mli));
            h5create(SavePath_H5_name, '/score_label', size(label), 'Datatype', 'single');
            h5write(SavePath_H5_name, '/score_label', single(label), [1,1], size(label));
            h5create(SavePath_H5_name, '/cls', size(cls), 'Datatype', 'single');
            h5write(SavePath_H5_name, '/cls', single(cls), [1,1], size(cls));
            h5create(SavePath_H5_name, '/level', size(level), 'Datatype', 'single');
            h5write(SavePath_H5_name, '/level', single(level), [1,1], size(level));
            idx = idx + 1;
        end
    end
    disp(['第 ', num2str(inum), ' 个场景生成', '运行时间: ',num2str(sprintf('%.3f', toc))]);
    inum = inum + 1;
end


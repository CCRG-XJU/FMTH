from Dataloader import *
import scipy.io as scio
from tqdm import tqdm
import numpy as np
import torch

def save_mat(query_c, gallery_c, query_y, gallery_y, save__dir,
             output_dim=64, map=0.0, dname='mir', test=1, l1=0.1, l2=0.2, mode_name="i2i"):
    save_dir = os.path.join(save__dir, "PR_cruve")
    os.makedirs(save_dir, exist_ok=True)
    query_c = query_c.cpu().detach().numpy()
    gallery_c = gallery_c.cpu().detach().numpy()
    query_y = query_y.cpu().detach().numpy()
    gallery_y = gallery_y.cpu().detach().numpy()

    result_dict = {
        'q_c': query_c,
        'g_c': gallery_c,
        'q_y': query_y,
        'g_y': gallery_y,
        'map': map
    }
    scio.savemat(os.path.join(save_dir, str(output_dim) + "-ours-" + dname + "-" + mode_name +"-test-" + str(test) +"-l1-" +str(l1) +"-l2-" +str(l2) + ".mat"), result_dict)
    print(f">>>>>> save best {mode_name} PR_cruve data!")

def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1]  # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH

def calculate_top_map(qB, rB, query_label, retrieval_label, topk):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param query_label: {0,1}^{mxl} query label
    :param retrieval_label: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    qB = qB.cpu().numpy()
    rB = rB.cpu().numpy()
    query_label = query_label.cpu().numpy()
    retrieval_label = retrieval_label.cpu().numpy()

    num_query = query_label.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(query_label[iter, :], retrieval_label.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def Evaluate_mAP(device, gallery_codes, query_codes, gallery_labels, query_labels, Top_N=None):
    mean_AP = calculate_top_map(qB=query_codes, rB=gallery_codes, query_label=query_labels, retrieval_label=gallery_labels, topk=Top_N)
    return mean_AP

def Evaluate_mAP2(device, gallery_codes, query_codes, gallery_labels, query_labels, Top_N=None):
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        retrieval = (query_labels[i, :] @ gallery_labels.t() > 0).float()
        hamming_dist = (gallery_codes.shape[1] - query_codes[i, :] @ gallery_codes.t())

        retrieval = retrieval[T.argsort(hamming_dist)][:Top_N]
        retrieval_cnt = retrieval.sum().int().item()

        if retrieval_cnt == 0:
            continue

        score = T.linspace(1, retrieval_cnt, retrieval_cnt).to(device, non_blocking=True)
        index = (T.nonzero(retrieval == 1, as_tuple=False).squeeze() + 1.0).float().to(device, non_blocking=True)

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP

def DoRetrieval_cifar(device, net, hash_layer, Y, Gallery_loader, Query_loader, NB_CLS, Top_N, args):
    sk_bits_list = list(map(int, args.N_bits.split(",")))  # str -> list
    gallery_c = dict()
    query_c = dict()
    mAPs = dict()
    Crop_Normalize = T.nn.Sequential(
        Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))
    )

    with T.no_grad():
        for i, data in tqdm(enumerate(Gallery_loader, 0)):
            gallery_x_batch, gallery_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            gallery_x_batch = Crop_Normalize(gallery_x_batch)
            if gallery_y_batch.dim() == 1: #对单标签网络进行one-hot编码
                gallery_y_batch = T.eye(NB_CLS, device=device)[gallery_y_batch]

            outputs = net(gallery_x_batch)
            Hcodes, _ = hash_layer(outputs, Y)

            if i == 0:
                for one in sk_bits_list:
                    gallery_c[str(one)] = Hcodes[str(one)]
                gallery_y = gallery_y_batch
            else:
                for one in sk_bits_list:
                    gallery_c[str(one)] = T.cat([gallery_c[str(one)], Hcodes[str(one)]], 0)
                gallery_y = T.cat([gallery_y, gallery_y_batch], 0)

        for i, data in tqdm(enumerate(Query_loader, 0)):
            query_x_batch, query_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            query_x_batch = Crop_Normalize(query_x_batch)
            if query_y_batch.dim() == 1:
                query_y_batch = T.eye(NB_CLS, device=device)[query_y_batch]

            outputs = net(query_x_batch)
            Hcodes, _ = hash_layer(outputs, Y)

            if i == 0:
                for one in sk_bits_list:
                    query_c[str(one)] = Hcodes[str(one)]
                query_y = query_y_batch
            else:
                for one in sk_bits_list:
                    query_c[str(one)] = T.cat([query_c[str(one)], Hcodes[str(one)]], 0)
                query_y = T.cat([query_y, query_y_batch], 0)

    # return mAPs
    for one in sk_bits_list:
        gallery_c[str(one)] = T.sign(gallery_c[str(one)])
        query_c[str(one)] = T.sign(query_c[str(one)])
        mAPs[str(one)] = Evaluate_mAP(device, gallery_c[str(one)], query_c[str(one)], gallery_y, query_y, Top_N)

    return mAPs, gallery_c, query_c, gallery_y, query_y

def DoRetrieval_imagenet(device, net, hash_layer, Y, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args):
    sk_bits_list = list(map(int, args.N_bits.split(",")))  # str -> list
    gallery_c = dict()
    query_c = dict()
    mAPs = dict()
    Gallery_set = Loader_imagenet(Img_dir, Gallery_dir, NB_CLS, data_set="test_set")
    Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    Query_set = Loader_imagenet(Img_dir, Query_dir, NB_CLS, data_set="test_set")
    Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    with T.no_grad():
        for i, data in tqdm(enumerate(Gallery_loader, 0)):
            gallery_x_batch, gallery_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            # gallery_x_batch = Crop_Normalize(gallery_x_batch)
            if gallery_y_batch.dim() == 1:
                gallery_y_batch = T.eye(NB_CLS, device=device)[gallery_y_batch]

            outputs = net(gallery_x_batch)
            Hcodes, _ = hash_layer(outputs, Y)

            if i == 0:
                for one in sk_bits_list:
                    gallery_c[str(one)] = Hcodes[str(one)]
                # gallery_c = outputs
                gallery_y = gallery_y_batch
            else:
                for one in sk_bits_list:
                    gallery_c[str(one)] = T.cat([gallery_c[str(one)], Hcodes[str(one)]], 0)
                # gallery_c = T.cat([gallery_c, outputs], 0)
                gallery_y = T.cat([gallery_y, gallery_y_batch], 0)

        for i, data in tqdm(enumerate(Query_loader, 0)):
            query_x_batch, query_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            # query_x_batch = Crop_Normalize(query_x_batch)
            if query_y_batch.dim() == 1:
                query_y_batch = T.eye(NB_CLS, device=device)[query_y_batch]

            outputs = net(query_x_batch)
            Hcodes, _ = hash_layer(outputs, Y)

            if i == 0:
                for one in sk_bits_list:
                    query_c[str(one)] = Hcodes[str(one)]
                query_y = query_y_batch
            else:
                for one in sk_bits_list:
                    query_c[str(one)] = T.cat([query_c[str(one)], Hcodes[str(one)]], 0)
                query_y = T.cat([query_y, query_y_batch], 0)

    # return mAPs
    for one in sk_bits_list:
        gallery_c[str(one)] = T.sign(gallery_c[str(one)])
        query_c[str(one)] = T.sign(query_c[str(one)])
        mAPs[str(one)] = Evaluate_mAP(device, gallery_c[str(one)], query_c[str(one)], gallery_y, query_y, Top_N)

    return mAPs, gallery_c, query_c, gallery_y, query_y

def DoRetrieval(device, Baseline, HashLayer, HP, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args):
    sk_bits_list = list(map(int, args.N_bits.split(",")))  # str -> list
    gallery_c = dict()
    query_c = dict()
    mAPs = dict()
    Gallery_set = Loader(Img_dir, Gallery_dir, NB_CLS)
    Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                             prefetch_factor=2)

    Query_set = Loader(Img_dir, Query_dir, NB_CLS)
    Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                           prefetch_factor=2)

    Crop_Normalize = T.nn.Sequential(
        Kg.CenterCrop(224),
        Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))
    )

    with T.no_grad():
        for i, data in tqdm(enumerate(Gallery_loader, 0)):
            gallery_x_batch, gallery_y_batch = data[0].to(device, non_blocking=True), data[1].to(device,
                                                                                                 non_blocking=True)
            gallery_x_batch = Crop_Normalize(gallery_x_batch)
            if gallery_y_batch.dim() == 1:
                gallery_y_batch = T.eye(NB_CLS, device=device)[gallery_y_batch]

            outputs = Baseline(gallery_x_batch)
            Hcodes, _ = HashLayer(outputs, HP)

            if i == 0:
                for one in sk_bits_list:
                    gallery_c[str(one)] = Hcodes[str(one)]
                gallery_y = gallery_y_batch
            else:
                for one in sk_bits_list:
                    gallery_c[str(one)] = T.cat([gallery_c[str(one)], Hcodes[str(one)]], 0)
                gallery_y = T.cat([gallery_y, gallery_y_batch], 0)

        for i, data in tqdm(enumerate(Query_loader, 0)):
            query_x_batch, query_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            query_x_batch = Crop_Normalize(query_x_batch)
            if query_y_batch.dim() == 1:
                query_y_batch = T.eye(NB_CLS, device=device)[query_y_batch]

            outputs = Baseline(query_x_batch)
            Hcodes, _ = HashLayer(outputs, HP)

            if i == 0:
                for one in sk_bits_list:
                    query_c[str(one)] = Hcodes[str(one)]
                query_y = query_y_batch
            else:
                for one in sk_bits_list:
                    query_c[str(one)] = T.cat([query_c[str(one)], Hcodes[str(one)]], 0)
                query_y = T.cat([query_y, query_y_batch], 0)

    for one in sk_bits_list:
        gallery_c[str(one)] = T.sign(gallery_c[str(one)])
        query_c[str(one)] = T.sign(query_c[str(one)])
        mAPs[str(one)] = Evaluate_mAP(device, gallery_c[str(one)], query_c[str(one)], gallery_y, query_y, Top_N)

    return mAPs, gallery_c, query_c, gallery_y, query_y
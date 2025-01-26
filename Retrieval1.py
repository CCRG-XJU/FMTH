from Dataloader import *
from Dataloader_2 import *
import scipy.io as scio
from tqdm import tqdm

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
        'map': map.cpu()
    }
    scio.savemat(os.path.join(save_dir, str(output_dim) + "-ours-" + dname + "-" + mode_name +"-test-" + str(test) +"-l1-" +str(l1) +"-l2-" +str(l2) + ".mat"), result_dict)
    print(f">>>>>> save best {mode_name} PR_cruve data!")

def Evaluate_mAP(device, gallery_codes, query_codes, gallery_labels, query_labels, Top_N=None):
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

def DoRetrieval_cifar(device, net, Gallery_loader, Query_loader, NB_CLS, Top_N, args):
    # Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    # Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    Crop_Normalize = T.nn.Sequential(
        # Kg.CenterCrop(32),
        Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))
    )

    with T.no_grad():
        for i, data in tqdm(enumerate(Gallery_loader, 0)):
            gallery_x_batch, gallery_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            gallery_x_batch = Crop_Normalize(gallery_x_batch)
            if gallery_y_batch.dim() == 1: #对单标签网络进行one-hot编码
                gallery_y_batch = T.eye(NB_CLS, device=device)[gallery_y_batch]

            outputs, _ = net(gallery_x_batch)

            if i == 0:
                gallery_c = outputs
                gallery_y = gallery_y_batch
            else:
                gallery_c = T.cat([gallery_c, outputs], 0)
                gallery_y = T.cat([gallery_y, gallery_y_batch], 0)

        for i, data in tqdm(enumerate(Query_loader, 0)):
            query_x_batch, query_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            query_x_batch = Crop_Normalize(query_x_batch)
            if query_y_batch.dim() == 1:
                query_y_batch = T.eye(NB_CLS, device=device)[query_y_batch]

            outputs, _ = net(query_x_batch)

            if i == 0:
                query_c = outputs
                query_y = query_y_batch
            else:
                query_c = T.cat([query_c, outputs], 0)
                query_y = T.cat([query_y, query_y_batch], 0)

    gallery_c = T.sign(gallery_c)
    query_c = T.sign(query_c)

    mAP = Evaluate_mAP(device, gallery_c, query_c, gallery_y, query_y, Top_N)
    return mAP, query_c, gallery_c, query_y, gallery_y

def DoRetrieval_cifar2(device, net, hash_layer, Y, Gallery_loader, Query_loader, NB_CLS, Top_N, args):
    # Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    # Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    Crop_Normalize = T.nn.Sequential(
        # Kg.CenterCrop(32),
        Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))
    )

    with T.no_grad():
        for i, data in tqdm(enumerate(Gallery_loader, 0)):
            gallery_x_batch, gallery_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            gallery_x_batch = Crop_Normalize(gallery_x_batch)
            if gallery_y_batch.dim() == 1: #对单标签网络进行one-hot编码
                gallery_y_batch = T.eye(NB_CLS, device=device)[gallery_y_batch]

            outputs = net(gallery_x_batch)
            outputs, _ = hash_layer(outputs, Y)

            if i == 0:
                gallery_c = outputs
                gallery_y = gallery_y_batch
            else:
                gallery_c = T.cat([gallery_c, outputs], 0)
                gallery_y = T.cat([gallery_y, gallery_y_batch], 0)

        for i, data in tqdm(enumerate(Query_loader, 0)):
            query_x_batch, query_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            query_x_batch = Crop_Normalize(query_x_batch)
            if query_y_batch.dim() == 1:
                query_y_batch = T.eye(NB_CLS, device=device)[query_y_batch]

            outputs = net(query_x_batch)
            outputs, _ = hash_layer(outputs, Y)

            if i == 0:
                query_c = outputs
                query_y = query_y_batch
            else:
                query_c = T.cat([query_c, outputs], 0)
                query_y = T.cat([query_y, query_y_batch], 0)

    gallery_c = T.sign(gallery_c)
    query_c = T.sign(query_c)

    mAP = Evaluate_mAP(device, gallery_c, query_c, gallery_y, query_y, Top_N)
    return mAP, query_c, gallery_c, query_y, gallery_y

def DoRetrieval_cifar3(device, net, hash_layer, Y, Gallery_loader, Query_loader, NB_CLS, Top_N, args):
    # Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    # Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)
    sk_bits_list = list(map(int, args.N_bits.split(",")))  # str -> list
    gallery_c = dict()
    query_c = dict()
    mAPs = dict()
    Crop_Normalize = T.nn.Sequential(
        # Kg.CenterCrop(32),
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

    # for one in sk_bits_list:
    #     gallery_c[str(one)] = T.sign(gallery_c[str(one)])
    #     query_c[str(one)] = T.sign(query_c[str(one)])
    #     mAPs[str(one)] = Evaluate_mAP(device, gallery_c[str(one)], query_c[str(one)], gallery_y, query_y, Top_N)

    # return mAPs
    for one in sk_bits_list:
        gallery_c[str(one)] = T.sign(gallery_c[str(one)])
        query_c[str(one)] = T.sign(query_c[str(one)])
        mAPs[str(one)] = Evaluate_mAP(device, gallery_c[str(one)], query_c[str(one)], gallery_y, query_y, Top_N)

    return mAPs, gallery_c, query_c, gallery_y, query_y

def DoRetrieval_imagenet(device, net, Img_query_dir, Img_database_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args):
    Gallery_set = Loader_imagenet(Img_database_dir, Gallery_dir, NB_CLS, args.dataset)
    Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    Query_set = Loader(Img_query_dir, Query_dir, NB_CLS, args.dataset)
    Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    Crop_Normalize = T.nn.Sequential(
        Kg.CenterCrop(224),
        Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))
    )

    with T.no_grad():
        for i, data in tqdm(enumerate(Gallery_loader, 0)):
            gallery_x_batch, gallery_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            gallery_x_batch = Crop_Normalize(gallery_x_batch)
            if gallery_y_batch.dim() == 1:
                gallery_y_batch = T.eye(NB_CLS, device=device)[gallery_y_batch]

            outputs, _ = net(gallery_x_batch)

            if i == 0:
                gallery_c = outputs
                gallery_y = gallery_y_batch
            else:
                gallery_c = T.cat([gallery_c, outputs], 0)
                gallery_y = T.cat([gallery_y, gallery_y_batch], 0)

        for i, data in tqdm(enumerate(Query_loader, 0)):
            query_x_batch, query_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            query_x_batch = Crop_Normalize(query_x_batch)
            if query_y_batch.dim() == 1:
                query_y_batch = T.eye(NB_CLS, device=device)[query_y_batch]

            outputs, _ = net(query_x_batch)

            if i == 0:
                query_c = outputs
                query_y = query_y_batch
            else:
                query_c = T.cat([query_c, outputs], 0)
                query_y = T.cat([query_y, query_y_batch], 0)

    gallery_c = T.sign(gallery_c)
    query_c = T.sign(query_c)

    mAP = Evaluate_mAP(device, gallery_c, query_c, gallery_y, query_y, Top_N)
    return mAP

def DoRetrieval_imagenet2(device, net, hash_layer, Y, Img_query_dir, Img_database_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args):
    Gallery_set = Loader3(Img_database_dir, Gallery_dir, NB_CLS, args.dataset)
    Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    Query_set = Loader3(Img_query_dir, Query_dir, NB_CLS, args.dataset)
    Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    # Crop_Normalize = T.nn.Sequential(
    #     Kg.CenterCrop(224),
    #     Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))
    # )

    with T.no_grad():
        for i, data in tqdm(enumerate(Gallery_loader, 0)):
            gallery_x_batch, gallery_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            # gallery_x_batch = Crop_Normalize(gallery_x_batch)
            if gallery_y_batch.dim() == 1:
                gallery_y_batch = T.eye(NB_CLS, device=device)[gallery_y_batch]

            outputs = net(gallery_x_batch)
            outputs, _ = hash_layer(outputs, Y)

            if i == 0:
                gallery_c = outputs
                gallery_y = gallery_y_batch
            else:
                gallery_c = T.cat([gallery_c, outputs], 0)
                gallery_y = T.cat([gallery_y, gallery_y_batch], 0)

        for i, data in tqdm(enumerate(Query_loader, 0)):
            query_x_batch, query_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            # query_x_batch = Crop_Normalize(query_x_batch)
            if query_y_batch.dim() == 1:
                query_y_batch = T.eye(NB_CLS, device=device)[query_y_batch]

            outputs = net(query_x_batch)
            outputs, _ = hash_layer(outputs, Y)

            if i == 0:
                query_c = outputs
                query_y = query_y_batch
            else:
                query_c = T.cat([query_c, outputs], 0)
                query_y = T.cat([query_y, query_y_batch], 0)

    gallery_c = T.sign(gallery_c)
    query_c = T.sign(query_c)

    mAP = Evaluate_mAP(device, gallery_c, query_c, gallery_y, query_y, Top_N)
    return mAP

def DoRetrieval_imagenet3(device, net, hash_layer, Y, Img_query_dir, Img_database_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args):
    sk_bits_list = list(map(int, args.N_bits.split(",")))  # str -> list
    gallery_c = dict()
    query_c = dict()
    mAPs = dict()
    Gallery_set = Loader3(Img_database_dir, Gallery_dir, NB_CLS, args.dataset)
    Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    Query_set = Loader3(Img_query_dir, Query_dir, NB_CLS, args.dataset)
    Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    # Crop_Normalize = T.nn.Sequential(
    #     Kg.CenterCrop(224),
    #     Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))
    # )

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

    # for one in sk_bits_list:
    #     gallery_c[str(one)] = T.sign(gallery_c[str(one)])
    #     query_c[str(one)] = T.sign(query_c[str(one)])
    #     mAPs[str(one)] = Evaluate_mAP(device, gallery_c[str(one)], query_c[str(one)], gallery_y, query_y, Top_N)

    # return mAPs
    for one in sk_bits_list:
        gallery_c[str(one)] = T.sign(gallery_c[str(one)])
        query_c[str(one)] = T.sign(query_c[str(one)])
        mAPs[str(one)] = Evaluate_mAP(device, gallery_c[str(one)], query_c[str(one)], gallery_y, query_y, Top_N)

    return mAPs, gallery_c, query_c, gallery_y, query_y


def DoRetrieval(device, net, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args):
    Gallery_set = Loader(Img_dir, Gallery_dir, NB_CLS, args.dataset)
    Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    Query_set = Loader(Img_dir, Query_dir, NB_CLS, args.dataset)
    Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    Crop_Normalize = T.nn.Sequential(
        Kg.CenterCrop(224),
        Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))
    )

    with T.no_grad():
        for i, data in tqdm(enumerate(Gallery_loader, 0)):
            gallery_x_batch, gallery_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            gallery_x_batch = Crop_Normalize(gallery_x_batch)
            if gallery_y_batch.dim() == 1:
                gallery_y_batch = T.eye(NB_CLS, device=device)[gallery_y_batch]

            outputs, _ = net(gallery_x_batch)

            if i == 0:
                gallery_c = outputs
                gallery_y = gallery_y_batch
            else:
                gallery_c = T.cat([gallery_c, outputs], 0)
                gallery_y = T.cat([gallery_y, gallery_y_batch], 0)

        for i, data in tqdm(enumerate(Query_loader, 0)):
            query_x_batch, query_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            query_x_batch = Crop_Normalize(query_x_batch)
            if query_y_batch.dim() == 1:
                query_y_batch = T.eye(NB_CLS, device=device)[query_y_batch]

            outputs, _ = net(query_x_batch)

            if i == 0:
                query_c = outputs
                query_y = query_y_batch
            else:
                query_c = T.cat([query_c, outputs], 0)
                query_y = T.cat([query_y, query_y_batch], 0)

    gallery_c = T.sign(gallery_c)
    query_c = T.sign(query_c)

    mAP = Evaluate_mAP(device, gallery_c, query_c, gallery_y, query_y, Top_N)
    return mAP

def DoRetrieval2(device, net, hash_layer, Y, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args):
    Gallery_set = Loader(Img_dir, Gallery_dir, NB_CLS, args.dataset)
    Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    Query_set = Loader(Img_dir, Query_dir, NB_CLS, args.dataset)
    Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    Crop_Normalize = T.nn.Sequential(
        Kg.CenterCrop(224),
        Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))
    )

    with T.no_grad():
        for i, data in tqdm(enumerate(Gallery_loader, 0)):
            gallery_x_batch, gallery_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            gallery_x_batch = Crop_Normalize(gallery_x_batch)
            if gallery_y_batch.dim() == 1:
                gallery_y_batch = T.eye(NB_CLS, device=device)[gallery_y_batch]

            outputs = net(gallery_x_batch)
            outputs, _ = hash_layer(outputs, Y)

            if i == 0:
                gallery_c = outputs
                gallery_y = gallery_y_batch
            else:
                gallery_c = T.cat([gallery_c, outputs], 0)
                gallery_y = T.cat([gallery_y, gallery_y_batch], 0)

        for i, data in tqdm(enumerate(Query_loader, 0)):
            query_x_batch, query_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            query_x_batch = Crop_Normalize(query_x_batch)
            if query_y_batch.dim() == 1:
                query_y_batch = T.eye(NB_CLS, device=device)[query_y_batch]

            outputs = net(query_x_batch)
            outputs, _ = hash_layer(outputs, Y)

            if i == 0:
                query_c = outputs
                query_y = query_y_batch
            else:
                query_c = T.cat([query_c, outputs], 0)
                query_y = T.cat([query_y, query_y_batch], 0)

    gallery_c = T.sign(gallery_c)
    query_c = T.sign(query_c)

    mAP = Evaluate_mAP(device, gallery_c, query_c, gallery_y, query_y, Top_N)
    return mAP

def DoRetrieval3(device, Baseline, HashLayer, HP, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args):
    sk_bits_list = list(map(int, args.N_bits.split(",")))  # str -> list
    gallery_c = dict()
    query_c = dict()
    mAPs = dict()
    Gallery_set = Loader(Img_dir, Gallery_dir, NB_CLS, args.dataset)
    Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                             prefetch_factor=2)

    Query_set = Loader(Img_dir, Query_dir, NB_CLS, args.dataset)
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

def DoRetrieval5(device, net, hash_layer, Y, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args):
    Gallery_set = Loader(Img_dir, Gallery_dir, NB_CLS, args.dataset)
    Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    Query_set = Loader(Img_dir, Query_dir, NB_CLS, args.dataset)
    Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=2)

    Crop_Normalize = T.nn.Sequential(
        Kg.CenterCrop(224),
        Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))
    )

    with T.no_grad():
        for i, data in tqdm(enumerate(Gallery_loader, 0)):
            gallery_x_batch, gallery_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            gallery_x_batch = Crop_Normalize(gallery_x_batch)
            if gallery_y_batch.dim() == 1:
                gallery_y_batch = T.eye(NB_CLS, device=device)[gallery_y_batch]

            outputs = net(gallery_x_batch)
            outputs, _ = hash_layer(outputs, Y)
            

            if i == 0:
                gallery_c = outputs
                gallery_y = gallery_y_batch
            else:
                gallery_c = T.cat([gallery_c, outputs], 0)
                gallery_y = T.cat([gallery_y, gallery_y_batch], 0)

        for i, data in tqdm(enumerate(Query_loader, 0)):
            query_x_batch, query_y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            query_x_batch = Crop_Normalize(query_x_batch)
            if query_y_batch.dim() == 1:
                query_y_batch = T.eye(NB_CLS, device=device)[query_y_batch]

            outputs = net(query_x_batch)
            outputs, _ = hash_layer(outputs, Y)

            if i == 0:
                query_c = outputs
                query_y = query_y_batch
            else:
                query_c = T.cat([query_c, outputs], 0)
                query_y = T.cat([query_y, query_y_batch], 0)

    gallery_c = T.sign(gallery_c)
    query_c = T.sign(query_c)
    # bit_scalable(device, hash_layer, gallery_c, query_c, gallery_y, query_y, Top_N, args, to_bit=[64, 32, 16])
    mAP = Evaluate_mAP(device, gallery_c, query_c, gallery_y, query_y, Top_N)
    return mAP, query_c, gallery_c, query_y, gallery_y
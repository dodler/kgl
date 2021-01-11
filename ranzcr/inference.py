def run_test(aug, ckpts, mdl):
    test_ds = CassavaDs(df=sub, aug=aug, path='../input/cassava-leaf-disease-classification/test_images')
    test_ld = torch.utils.data.DataLoader(test_ds, shuffle=False, num_workers=2, batch_size=32)

    for checkpoint_path in ckpts:
        print('doing checkpoint', checkpoint_path)
        ckpt=torch.load(checkpoint_path, map_location='cpu')
        print(mdl.load_state_dict(ckpt))

        pred_labels = []
        for x,_ in tqdm(test_ld):
            x_base=x.to(device)

            x=x_base
            p1=mdl(x).cpu().numpy()

            pred=p1
            pred_labels.append(pred)
    pred_labels = np.concatenate(pred_labels)
    return pred_labels
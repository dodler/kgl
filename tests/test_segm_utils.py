def predict_df(df, n_samples=-1, shuffle=False):
    if n_samples > 0:
        df = df.sample(n_samples).reset_index().drop('index', axis=1)
    n = df.shape[0]

    result = []
    imgs = []
    gt_masks = []

    for i in tqdm(range(n)):

        m = df.iloc[idx, 1]
        if isinstance(m, str):
            m = rle2mask(m)
        else:
            m = np.zeros((256, 1600), np.uint8)
        gt_masks.append(m)

        img = df.iloc[i, 0].split('_')[0]
        p = osp.join('/var/ssd_1t/severstal/train/', img)
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img.copy())

        pred = predict_image(p, model)
        result.append(pred)

    return imgs, result, gt_masks
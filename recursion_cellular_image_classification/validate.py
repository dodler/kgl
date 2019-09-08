def make_predicts(loader, model_ft):
    result_labels = []
    result_preds = []
    for data in tqdm(loader):
        data_input, target = data
        data_input = data_input.cuda()
        target = target.cuda()
        result_labels.append(target.detach().cpu().numpy())
        with torch.no_grad():
            pred = model_ft(data_input, None).detach().cpu().numpy()
            result_preds.append(pred)

    return np.concatenate(result_labels), np.concatenate(result_preds)

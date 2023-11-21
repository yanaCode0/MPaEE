import copy
import wandb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from networks import _create_vision_transformer, prompt_triple_class

def training(args, prototype_mode, taskid, train_loader, test_loader, known_classes=0, vitprompt=None, numclass=10):
    trainer = pl.Trainer(default_root_dir='./checkpoints/'+wandb.run.name, accelerator="gpu", devices=1,logger=True,
                         max_epochs=args.max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="train_acc"),
                                    LearningRateMonitor("epoch"),])

    if taskid == 0:
        model = prototype_mode(num_cls=numclass, lr=args.lr, max_epoch=args.max_epochs, weight_decay=0.0005,
                        known_classes=known_classes, freezep=False, using_prompt=args.using_prompt,
                        anchor_energy=args.anchor_energy, lamda=args.lamda, energy_beta=args.energy_beta)
    else:
        model = prototype_mode(num_cls=numclass, lr=args.lr, max_epoch=args.max_epochs, weight_decay=0.0005,
                        known_classes=known_classes, freezep=True, using_prompt=args.using_prompt,
                               anchor_energy=args.anchor_energy, lamda=args.lamda, energy_beta=args.energy_beta)
        model.vitprompt = vitprompt

    trainer.fit(model, train_loader)
    if args.dataset == "core50":
        val_result = [{'test_acc':0}]
    else:
        val_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(val_result)
    return model, val_result

def training_prompt(args, prototype_mode, taskid, train_loader, test_loader, known_classes=0, vitprompt=None, numclass=10, pre_prompt=None, cls_inc=10):
    trainer = pl.Trainer(default_root_dir='./checkpoints/'+wandb.run.name, accelerator="gpu", devices=1,logger=True,
                         max_epochs=args.max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="train_acc"),
                                    LearningRateMonitor("epoch"),])

    if taskid == 0:
        model = prototype_mode(num_cls=numclass, lr=args.lr, max_epoch=args.max_epochs, weight_decay=0.0005,
                        known_classes=known_classes, freezep=False, using_prompt=args.using_prompt,
                        anchor_energy=args.anchor_energy, lamda=args.lamda, energy_beta=args.energy_beta,cls_inc=numclass)
    else:
        model = prototype_mode(num_cls=numclass, lr=args.lr, max_epoch=args.max_epochs, weight_decay=0.0005,
                        known_classes=known_classes, freezep=True, using_prompt=args.using_prompt,
                        anchor_energy=args.anchor_energy, lamda=args.lamda, energy_beta=args.energy_beta,cls_inc=cls_inc)
        model.pre_prompt = torch.cat(pre_prompt, dim=0)
        model.pre_prompt.requires_grad=False
        model.vitprompt=vitprompt

    #model.pre_train_process(train_loader)
    trainer.fit(model, train_loader)
    if args.dataset == "core50":
        val_result = [{'test_acc':0}]
    else:
        val_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(val_result)
    return model, val_result


def eval(args, load_path, datamanage):
    assembles = torch.load(load_path, map_location=torch.device('cpu'))
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    ptvit = _create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)

    device = 'cuda:0'
    ptvit = ptvit.to(device)
    ptvit.eval()
    all_tabs = assembles['all_tabs']
    all_classifiers = assembles['all_classifiers']
    all_tokens = assembles['all_tokens']
    vitpromptlist = assembles['vitpromptlist']

    all_tabs = [i.to(device) for i in all_tabs]
    all_classifiers = [i.to(device) for i in all_classifiers]
    all_tokens = [i.to(device) for i in all_tokens]
    vitpromptlist = [i.to(device) for i in vitpromptlist]

    _known_classes=0
    # fast mode
    candidata_temperatures = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # slow mode
    # candidata_temperatures = [i/1000 for i in range(1,1000,1)]
    select_temperature = [0.001] # for first stage with no former data
    accuracy_table, accs, t_accs = [], [], []
    print("Testing...")
    for taskid in range(datamanage.nb_tasks):
        _total_classes = _known_classes + datamanage.get_task_size(taskid)
        test_till_now_dataset = datamanage.get_dataset(np.arange(0, _total_classes), source='test', mode='test')
        test_till_now_loader = DataLoader(test_till_now_dataset, batch_size=32, shuffle=False, num_workers=1)

        if taskid > 0:
            classifiers = all_classifiers[:taskid+1]
            task_tokens = all_tokens[:taskid+1]
            tabs = all_tabs[:taskid+1]
            current_dataset = datamanage.get_dataset(np.arange(_known_classes, _total_classes), source='train', mode='test')
            current_dataloader = DataLoader(current_dataset, batch_size=32, shuffle=False, num_workers=1)
            all_energies = {i: [] for i in candidata_temperatures}
            with torch.no_grad():
                for _, (_, inputs, targets) in enumerate(current_dataloader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    energys = {i: [] for i in candidata_temperatures}
                    image_features = ptvit(inputs, instance_tokens=vitpromptlist[taskid].weight, returnbeforepool=True)
                    # image_features = ptvit(inputs, returnbeforepool=True)

                    B = image_features.shape[0]
                    for idx, fc in enumerate(classifiers):
                        task_token = task_tokens[idx].expand(B, -1, -1)
                        task_token, attn, v = tabs[idx](torch.cat((task_token, image_features), dim=1), mask_heads=None)
                        task_token = task_token[:, 0]
                        logit = fc(task_token)

                        for tem in candidata_temperatures:
                            energys[tem].append(torch.logsumexp(logit / tem, axis=-1))
                    energys = {i: torch.stack(energys[i]).T for i in candidata_temperatures}
                    for i in candidata_temperatures:
                        all_energies[i].append(energys[i])

            all_energies = {i: torch.cat(all_energies[i]) for i in candidata_temperatures}
            seperation_accuracy = []
            for i in candidata_temperatures:
                seperation_accuracy.append((sum(all_energies[i].max(1)[1]==(taskid))/len(all_energies[i])).item())
            select_temperature.append(candidata_temperatures[np.array(seperation_accuracy).argmax()])

        set_select_temperature = list(set(select_temperature))
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(test_till_now_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            candiatetask = {i: [] for i in set_select_temperature}
            seperatePreds = []

            with torch.no_grad():
                image_features = ptvit(inputs, instance_tokens=vitpromptlist[taskid].weight, returnbeforepool=True)
                # image_features = ptvit(inputs, returnbeforepool=True)
                B = image_features.shape[0]
                for idx, fc in enumerate(all_classifiers[:taskid+1]):
                    task_token = all_tokens[:taskid+1][idx].expand(B, -1, -1)
                    task_token, attn, v = all_tabs[:taskid+1][idx](torch.cat((task_token, image_features), dim=1), mask_heads=None)
                    task_token = task_token[:, 0]
                    logit = fc(task_token)
                    for tem in set_select_temperature:
                        candiatetask[tem].append(torch.logsumexp(logit / tem, axis=-1))
                    seperatePreds.append(logit.max(1)[1]+idx*logit.shape[1])

            candiatetask = {i: torch.stack(candiatetask[i]).T  for i in set_select_temperature}
            seperatePreds = torch.stack(seperatePreds).T

            pred = []
            for tem in set_select_temperature:
                val, ind = candiatetask[tem].max(1)
                pred.append(ind)
            indexselection = torch.stack(pred, 1)
            selectid = torch.mode(indexselection, dim=1, keepdim=False)[0]
            outputs = []
            for row, idx in enumerate(selectid):
                outputs.append(seperatePreds[row][idx])
            outputs = torch.stack(outputs)
            y_pred.append(outputs.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        if args.dil:
            accs.append(np.around((y_pred.T%args.max_cls == y_true%args.max_cls).sum() * 100 / len(y_true), decimals=2))
        else:
            accs.append(np.around((y_pred.T == y_true).sum() * 100 / len(y_true), decimals=2))
        t_accs.append(np.around((y_pred.T//args.max_cls == taskid).sum() * 100 / len(y_true), decimals=2))

        tempacc = []
        for class_id in range(0, np.max(y_true), _total_classes-_known_classes):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + _total_classes-_known_classes))[0]
            tempacc.append(np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=3))
        accuracy_table.append(tempacc)

        _known_classes = _total_classes


    np_acctable = np.zeros([taskid + 1, taskid + 1])
    for idxx, line in enumerate(accuracy_table):
        idxy = len(line)
        np_acctable[idxx, :idxy] = np.array(line)
    # import pdb;pdb.set_trace()
    np_acctable = np_acctable.T
    print("Accuracy table:")
    print(np_acctable)
    print("Accuracy curve:")
    print(accs)
    print("FAA: {}".format(accs[-1]))
    forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, taskid])[:taskid])
    print("FF: {}".format(forgetting))
    print('task acc:')
    print(t_accs)

def eval_prompt(args, load_path, datamanage):
    assembles = torch.load(load_path, map_location=torch.device('cpu'))
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    ptvit = _create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)

    device = 'cuda:0'
    ptvit = ptvit.to(device)
    ptvit.eval()
    all_tabs = assembles['all_tabs']
    all_classifiers = assembles['all_classifiers']
    all_tokens = assembles['all_tokens']
    vitpromptlist = assembles['vitpromptlist']
    vit_promptlist = assembles['prompts']

    all_tabs = [i.to(device) for i in all_tabs]
    all_classifiers = [i.to(device) for i in all_classifiers]
    all_tokens = [i.to(device) for i in all_tokens]
    vitpromptlist = [i.to(device) for i in vitpromptlist]
    vit_promptlist = [i.to(device) for i in vit_promptlist]
    prompts = torch.cat(vit_promptlist,dim=0)

    _known_classes=0
    # fast mode
    candidata_temperatures = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # slow mode
    # candidata_temperatures = [i/1000 for i in range(1,1000,1)]
    select_temperature = [0.001] # for first stage with no former data
    accuracy_table, accs = [], []
    print("Testing...")
    for taskid in range(datamanage.nb_tasks):
        _total_classes = _known_classes + datamanage.get_task_size(taskid)
        test_till_now_dataset = datamanage.get_dataset(np.arange(0, _total_classes), source='test', mode='test')
        test_till_now_loader = DataLoader(test_till_now_dataset, batch_size=32, shuffle=False, num_workers=1)

        if taskid > 0:
            classifiers = all_classifiers[:taskid+1]
            task_tokens = all_tokens[:taskid+1]
            tabs = all_tabs[:taskid+1]
            current_dataset = datamanage.get_dataset(np.arange(_known_classes, _total_classes), source='train', mode='test')
            current_dataloader = DataLoader(current_dataset, batch_size=32, shuffle=False, num_workers=1)
            all_energies = {i: [] for i in candidata_temperatures}
            with torch.no_grad():
                for _, (_, inputs, targets) in enumerate(current_dataloader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    energys = {i: [] for i in candidata_temperatures}
                    image_features = ptvit(inputs, instance_tokens=vitpromptlist[taskid].weight, returnbeforepool=True)
                    # image_features = ptvit(inputs, returnbeforepool=True)
                    prompt, s, pre_s, inter_s, total = prompt_triple_class(torch.mean(image_features,dim=1), prompts, infer=True, d=ptvit.embed_dim)

                    B = image_features.shape[0]
                    for idx, fc in enumerate(classifiers):
                        task_token = task_tokens[idx].expand(B, -1, -1)
                        task_token, attn, v = tabs[idx](torch.cat([task_token, prompt, image_features], dim=1), mask_heads=None)
                        task_token = task_token[:, 0]
                        logit = fc(task_token) * total[:,idx*10:(idx+1)*10]

                        for tem in candidata_temperatures:
                            energys[tem].append(torch.logsumexp(logit / tem, axis=-1))
                    energys = {i: torch.stack(energys[i]).T for i in candidata_temperatures}
                    for i in candidata_temperatures:
                        all_energies[i].append(energys[i])

            all_energies = {i: torch.cat(all_energies[i]) for i in candidata_temperatures}
            seperation_accuracy = []
            for i in candidata_temperatures:
                seperation_accuracy.append((sum(all_energies[i].max(1)[1]==(taskid))/len(all_energies[i])).item())
            select_temperature.append(candidata_temperatures[np.array(seperation_accuracy).argmax()])

        set_select_temperature = list(set(select_temperature))
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(test_till_now_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            candiatetask = {i: [] for i in set_select_temperature}
            seperatePreds = []

            with torch.no_grad():
                image_features = ptvit(inputs, instance_tokens=vitpromptlist[taskid].weight, returnbeforepool=True)
                # image_features = ptvit(inputs, returnbeforepool=True)
                prompt, s, pre_s, inter_s, total = prompt_triple_class(torch.mean(image_features,dim=1), prompts, infer=True, d=ptvit.embed_dim)
                B = image_features.shape[0]
                for idx, fc in enumerate(all_classifiers[:taskid+1]):
                    task_token = all_tokens[:taskid+1][idx].expand(B, -1, -1)
                    task_token, attn, v = all_tabs[:taskid+1][idx](torch.cat([task_token, prompt, image_features], dim=1), mask_heads=None)
                    task_token = task_token[:, 0]
                    logit = fc(task_token) * total[:,idx*10:(idx+1)*10]
                    for tem in set_select_temperature:
                        candiatetask[tem].append(torch.logsumexp(logit / tem, axis=-1))
                    seperatePreds.append(logit.max(1)[1]+idx*logit.shape[1])

            candiatetask = {i: torch.stack(candiatetask[i]).T  for i in set_select_temperature}
            seperatePreds = torch.stack(seperatePreds).T

            pred = []
            for tem in set_select_temperature:
                val, ind = candiatetask[tem].max(1)
                pred.append(ind)
            indexselection = torch.stack(pred, 1)
            selectid = torch.mode(indexselection, dim=1, keepdim=False)[0]
            outputs = []
            for row, idx in enumerate(selectid):
                outputs.append(seperatePreds[row][idx])
            outputs = torch.stack(outputs)
            y_pred.append(outputs.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        if args.dil:
            accs.append(np.around((y_pred.T%args.max_cls == y_true%args.max_cls).sum() * 100 / len(y_true), decimals=2))
        else:
            accs.append(np.around((y_pred.T == y_true).sum() * 100 / len(y_true), decimals=2))

        tempacc = []
        for class_id in range(0, np.max(y_true), _total_classes-_known_classes):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + _total_classes-_known_classes))[0]
            tempacc.append(np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=3))
        accuracy_table.append(tempacc)

        _known_classes = _total_classes


    np_acctable = np.zeros([taskid + 1, taskid + 1])
    for idxx, line in enumerate(accuracy_table):
        idxy = len(line)
        np_acctable[idxx, :idxy] = np.array(line)
    # import pdb;pdb.set_trace()
    np_acctable = np_acctable.T
    print("Accuracy table:")
    print(np_acctable)
    print("Accuracy curve:")
    print(accs)
    print("FAA: {}".format(accs[-1]))

    forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, taskid])[:taskid])
    print("FF: {}".format(forgetting))

def eval_2(args, load_path, datamanage, cls_inc=10):
    assembles = torch.load(load_path, map_location=torch.device('cpu'))
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    ptvit = _create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)

    device = 'cuda:0'
    ptvit = ptvit.to(device)
    ptvit.eval()
    all_tabs = assembles['all_tabs']
    all_classifiers = assembles['all_classifiers']
    all_tokens = assembles['all_tokens']
    vitpromptlist = assembles['vitpromptlist']
    vit_promptlist = assembles['prompts']

    all_tabs = [i.to(device) for i in all_tabs]
    all_classifiers = [i.to(device) for i in all_classifiers]
    all_tokens = [i.to(device) for i in all_tokens]
    vitpromptlist = [i.to(device) for i in vitpromptlist]
    vit_promptlist = [i.to(device) for i in vit_promptlist]
    prompts = torch.cat(vit_promptlist,dim=0)
    T = 0.1
    if args.dataset == 'core50' or args.dataset == 'domainnet':
        T = 0.5

    _known_classes=0
    init_cls=datamanage.get_task_size(0)

    # fast mode
    accuracy_table, accs, p_accs, p_table, t_accs = [], [], [], [], []
    print("Testing...")
    print(datamanage.nb_tasks)
    for taskid in range(datamanage.nb_tasks):
        _total_classes = _known_classes + datamanage.get_task_size(taskid)
        test_till_now_dataset = datamanage.get_dataset(np.arange(0, _total_classes), source='test', mode='test')
        test_till_now_loader = DataLoader(test_till_now_dataset, batch_size=32, shuffle=False, num_workers=1)
        y_pred, y_true, prompt_pred = [], [], []
        for _, (_, inputs, targets) in enumerate(test_till_now_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            seperatePreds = []

            with torch.no_grad():
                image_features = ptvit(inputs, instance_tokens=vitpromptlist[taskid].weight, returnbeforepool=True)
                # image_features = ptvit(inputs, returnbeforepool=True)
                prompt, s, pre_s, inter_s, total = prompt_triple_class(torch.mean(image_features,dim=1), prompts, infer=True, d=ptvit.embed_dim, cls_inc=_total_classes-_known_classes)
                tem_s = torch.nn.Softmax(-1)(total/T)
                prompt_pred.append(total[:,:init_cls+taskid*cls_inc].argmax(-1).cpu().numpy())
                B = image_features.shape[0]
                for idx, fc in enumerate(all_classifiers[:taskid+1]):
                    task_token = all_tokens[:taskid+1][idx].expand(B, -1, -1)
                    task_token, attn, v = all_tabs[:taskid+1][idx](torch.cat([task_token, prompt, image_features], dim=1), mask_heads=None)
                    task_token = task_token[:, 0]
                    #tem_s = total[:,idx*10:(idx+1)*10]
                    if idx==0:
                        logit = fc(task_token) * tem_s[:,0:init_cls]
                    else:
                        logit = fc(task_token) * tem_s[:,init_cls+(idx-1)*cls_inc:init_cls+idx*cls_inc]
                    #logit = fc(task_token)
                    seperatePreds.append(logit)

            outputs = []
            pred = torch.cat(seperatePreds,dim=-1)
            outputs = pred.argmax(-1).T
            y_pred.append(outputs.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        prompt_pred = np.concatenate(prompt_pred)
        # print(y_pred.T == y_true)
        if args.dil:
            accs.append(np.around((y_pred.T%args.max_cls == y_true%args.max_cls).sum() * 100 / len(y_true), decimals=2))
            p_accs.append(np.around((prompt_pred.T%args.init_cls == y_true%args.init_cls).sum() * 100 / len(y_true), decimals=2))
            t_accs.append(np.around((y_pred.T//args.max_cls == taskid).sum() * 100 / len(y_true), decimals=2))
        else:
            accs.append(np.around((y_pred.T == y_true).sum() * 100 / len(y_true), decimals=2))
            p_accs.append(np.around((prompt_pred.T//args.init_cls == y_true//args.init_cls).sum() * 100 / len(y_true), decimals=2))
            t_accs.append(np.around((y_pred.T//args.init_cls == y_true//args.init_cls).sum() * 100 / len(y_true), decimals=2))


        tempacc = []
        p_tempacc = []
        for class_id in range(0, np.max(y_true), _total_classes-_known_classes):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + _total_classes-_known_classes))[0]
            tempacc.append(np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=3))
            p_tempacc.append(np.around((prompt_pred[idxes]//args.init_cls == y_true[idxes]//args.init_cls).sum() * 100 / len(idxes), decimals=3))

        accuracy_table.append(tempacc)
        p_table.append(p_tempacc)

        _known_classes = _total_classes


    np_acctable = np.zeros([taskid + 1, taskid + 1])
    for idxx, line in enumerate(accuracy_table):
        idxy = len(line)
        np_acctable[idxx, :idxy] = np.array(line)
    np_acctable = np_acctable.T

    p_acctable = np.zeros([taskid + 1, taskid + 1])
    for idxx, line in enumerate(p_table):
        idxy = len(line)
        p_acctable[idxx, :idxy] = np.array(line)
    ## import pdb;pdb.set_trace()
    p_acctable = p_acctable.T
    print("Accuracy table:")
    print(np_acctable)
    print("Accuracy curve:")
    print(accs)
    print("FAA: {}".format(accs[-1]))
    forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, taskid])[:taskid])
    print("FF: {}".format(forgetting))
    #print("prompt acc:")
    #print(p_acctable)
    #print(p_accs)
    #print('task acc:')
    #print(t_accs)

    #visul_prompt_0 = visul_prompt[:,0,:].view(datamanage.nb_tasks,-1)
    #visul_prompt_1 = visul_prompt[:,1,:].view(datamanage.nb_tasks,-1)
    #print("prompt similarity of different tasks:")
    #print(torch.nn.CosineSimilarity(dim=-1)(visul_prompt_0.unsqueeze(1), visul_prompt_1))

def eval_3(args, load_path, datamanage):
    assembles = torch.load(load_path, map_location=torch.device('cpu'))
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    ptvit = _create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)

    device = 'cuda:0'
    ptvit = ptvit.to(device)
    ptvit.eval()
    all_tabs = assembles['all_tabs']
    all_classifiers = assembles['all_classifiers']
    all_tokens = assembles['all_tokens']
    vitpromptlist = assembles['vitpromptlist']

    all_tabs = [i.to(device) for i in all_tabs]
    all_classifiers = [i.to(device) for i in all_classifiers]
    all_tokens = [i.to(device) for i in all_tokens]
    vitpromptlist = [i.to(device) for i in vitpromptlist]

    _known_classes=0
    # fast mode
    accuracy_table, accs = [], []
    print("Testing...")
    for taskid in range(datamanage.nb_tasks):
        _total_classes = _known_classes + datamanage.get_task_size(taskid)
        test_till_now_dataset = datamanage.get_dataset(np.arange(0, _total_classes), source='test', mode='test')
        test_till_now_loader = DataLoader(test_till_now_dataset, batch_size=32, shuffle=False, num_workers=1)
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(test_till_now_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            seperatePreds = []

            with torch.no_grad():
                image_features = ptvit(inputs, instance_tokens=vitpromptlist[taskid].weight, returnbeforepool=True)
                B = image_features.shape[0]
                for idx, fc in enumerate(all_classifiers[:taskid+1]):
                    task_token = all_tokens[:taskid+1][idx].expand(B, -1, -1)
                    task_token, attn, v = all_tabs[:taskid+1][idx](torch.cat([task_token, image_features], dim=1), mask_heads=None)
                    task_token = task_token[:, 0]
                    logit = fc(task_token)
                    seperatePreds.append(logit)

            outputs = []
            pred = torch.cat(seperatePreds,dim=-1)
            outputs = pred.argmax(-1).T
            y_pred.append(outputs.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        # print(y_pred.T == y_true)
        # print((y_pred.T == y_true).sum())
        if args.dil:
            accs.append(np.around((y_pred.T%args.max_cls == y_true%args.max_cls).sum() * 100 / len(y_true), decimals=2))
        else:
            accs.append(np.around((y_pred.T == y_true).sum() * 100 / len(y_true), decimals=2))

        tempacc = []
        for class_id in range(0, np.max(y_true), _total_classes-_known_classes):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + _total_classes-_known_classes))[0]
            tempacc.append(np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=3))
        accuracy_table.append(tempacc)

        _known_classes = _total_classes


    np_acctable = np.zeros([taskid + 1, taskid + 1])
    for idxx, line in enumerate(accuracy_table):
        idxy = len(line)
        np_acctable[idxx, :idxy] = np.array(line)
    # import pdb;pdb.set_trace()
    np_acctable = np_acctable.T
    print("Accuracy table:")
    print(np_acctable)
    print("Accuracy curve:")
    print(accs)
    print("FAA: {}".format(accs[-1]))
    print()
    #print(np.max(np_acctable, axis=1) - np_acctable[:, taskid])
    forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, taskid])[:taskid])
    print("FF: {}".format(forgetting))


def eval_4(args, load_path, datamanage, cls_inc=10):
    assembles = torch.load(load_path, map_location=torch.device('cpu'))
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    ptvit = _create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)

    device = 'cuda:0'
    ptvit = ptvit.to(device)
    ptvit.eval()
    all_tabs = assembles['all_tabs']
    all_classifiers = assembles['all_classifiers']
    all_tokens = assembles['all_tokens']
    vitpromptlist = assembles['vitpromptlist']
    vit_promptlist = assembles['prompts']

    all_tabs = [i.to(device) for i in all_tabs]
    all_classifiers = [i.to(device) for i in all_classifiers]
    all_tokens = [i.to(device) for i in all_tokens]
    vitpromptlist = [i.to(device) for i in vitpromptlist]
    vit_promptlist = [i.to(device) for i in vit_promptlist]
    prompts = torch.cat(vit_promptlist,dim=0)
    for i in all_classifiers:
        print(i.weight.shape)

    _known_classes=0
    init_cls=datamanage.get_task_size(0)

    # fast mode
    accuracy_table, accs, p_accs, p_table, t_accs = [], [], [], [], []
    print("Testing...")
    for taskid in range(datamanage.nb_tasks):
        _total_classes = _known_classes + datamanage.get_task_size(taskid)
        test_till_now_dataset = datamanage.get_dataset(np.arange(0, _total_classes), source='test', mode='test')
        test_till_now_loader = DataLoader(test_till_now_dataset, batch_size=32, shuffle=False, num_workers=1)
        y_pred, y_true, prompt_pred = [], [], []
        for _, (_, inputs, targets) in enumerate(test_till_now_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            seperatePreds = []

            with torch.no_grad():
                image_features = ptvit(inputs, instance_tokens=vitpromptlist[taskid].weight, returnbeforepool=True)
                # image_features = ptvit(inputs, returnbeforepool=True)
                prompt, s, pre_s, inter_s, total = prompt_triple_class(torch.mean(image_features,dim=1), prompts, infer=True, d=ptvit.embed_dim, cls_inc=_total_classes-_known_classes)
                tem_s = torch.nn.Softmax(-1)(total/0.1)
                prompt_pred.append(total.view(targets.size(0),-1, 10).mean(-1)[:,:taskid+1].argmax(-1).cpu().numpy())
                B = image_features.shape[0]
                for idx, fc in enumerate(all_classifiers[:taskid+1]):
                    task_token = all_tokens[:taskid+1][idx].expand(B, -1, -1)
                    task_token, attn, v = all_tabs[:taskid+1][idx](torch.cat([task_token, prompt, image_features], dim=1), mask_heads=None)
                    task_token = task_token[:, 0]
                    #tem_s = total[:,idx*10:(idx+1)*10]
                    if idx==0:
                        logit = fc(task_token) * torch.mean(tem_s[:,0],-1).view(-1,1)
                    else:
                        logit = fc(task_token) * torch.mean(tem_s[:,idx],-1).view(-1,1)
                    #logit = fc(task_token)


                    seperatePreds.append(logit)

            outputs = []
            pred = torch.cat(seperatePreds,dim=-1)
            outputs = pred.argmax(-1).T
            y_pred.append(outputs.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        prompt_pred = np.concatenate(prompt_pred)
        # print(y_pred.T == y_true)
        if args.dil:
            accs.append(np.around((y_pred.T%args.max_cls == y_true%args.max_cls).sum() * 100 / len(y_true), decimals=2))
        else:
            accs.append(np.around((y_pred.T == y_true).sum() * 100 / len(y_true), decimals=2))
        p_accs.append(np.around((prompt_pred.T == y_true//args.init_cls).sum() * 100 / len(y_true), decimals=2))
        t_accs.append(np.around((y_pred.T//args.init_cls == y_true//args.init_cls).sum() * 100 / len(y_true), decimals=2))

        tempacc = []
        p_tempacc = []
        for class_id in range(0, np.max(y_true), _total_classes-_known_classes):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + _total_classes-_known_classes))[0]
            tempacc.append(np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=3))
            p_tempacc.append(np.around((prompt_pred[idxes] == y_true[idxes]//args.init_cls).sum() * 100 / len(idxes), decimals=3))

        accuracy_table.append(tempacc)
        p_table.append(p_tempacc)

        _known_classes = _total_classes


    np_acctable = np.zeros([taskid + 1, taskid + 1])
    for idxx, line in enumerate(accuracy_table):
        idxy = len(line)
        np_acctable[idxx, :idxy] = np.array(line)
    np_acctable = np_acctable.T

    p_acctable = np.zeros([taskid + 1, taskid + 1])
    for idxx, line in enumerate(p_table):
        idxy = len(line)
        p_acctable[idxx, :idxy] = np.array(line)
    ## import pdb;pdb.set_trace()
    p_acctable = p_acctable.T
    print("Accuracy table:")
    print(np_acctable)
    print("Accuracy curve:")
    print(accs)
    print("FAA: {}".format(accs[-1]))
    forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, taskid])[:taskid])
    print("FF: {}".format(forgetting))
    print("prompt acc:")
    print(p_acctable)
    print(p_accs)
    print('task acc:')
    print(t_accs)


def eval_5(args, load_path, datamanage, cls_inc=10):
    assembles = torch.load(load_path, map_location=torch.device('cpu'))
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    ptvit = _create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)

    device = 'cuda:0'
    ptvit = ptvit.to(device)
    ptvit.eval()
    all_tabs = assembles['all_tabs']
    all_classifiers = assembles['all_classifiers']
    all_tokens = assembles['all_tokens']
    vitpromptlist = assembles['vitpromptlist']
    vit_promptlist = assembles['prompts']

    all_tabs = [i.to(device) for i in all_tabs]
    all_classifiers = [i.to(device) for i in all_classifiers]
    all_tokens = [i.to(device) for i in all_tokens]
    vitpromptlist = [i.to(device) for i in vitpromptlist]
    vit_promptlist = [i.to(device) for i in vit_promptlist]
    prompts = torch.cat(vit_promptlist,dim=0)



    #score_all = torch.nn.CosineSimilarity(dim=-1)(prompts.unsqueeze(1), prompts)
    #score_all = score_all.view(datamanage.nb_tasks,datamanage.nb_tasks,args.init_cls,args.init_cls).mean(-1).mean(1).view(datamanage.nb_tasks,-1)
    #print(score_all)
    #for i in all_classifiers:
        #print(i.weight.shape)

    _known_classes=0
    init_cls=datamanage.get_task_size(0)

    # fast mode
    accuracy_table, accs, p_accs, p_table = [], [], [], []
    print("Testing...")
    for taskid in range(datamanage.nb_tasks):
        _total_classes = _known_classes + datamanage.get_task_size(taskid)
        test_till_now_dataset = datamanage.get_dataset(np.arange(0, _total_classes), source='test', mode='test')
        test_till_now_loader = DataLoader(test_till_now_dataset, batch_size=32, shuffle=False, num_workers=1)
        y_pred, y_true, prompt_pred = [], [], []
        for _, (_, inputs, targets) in enumerate(test_till_now_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            seperatePreds = []

            with torch.no_grad():
                x = ptvit.patch_embed(inputs)
                x = torch.cat((ptvit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                x = x + ptvit.pos_embed.to(x.dtype)
                prompt, s, pre_s, inter_s, total = prompt_triple_class(torch.mean(x,dim=1), prompts, infer=True, d=ptvit.embed_dim, cls_inc=_total_classes-_known_classes)
                x = torch.cat([x[:,:1,:],prompt, x[:,1:,:]], dim=1)
                image_features = ptvit(x, instance_tokens=vitpromptlist[taskid].weight, returnbeforepool=True)
                # image_features = ptvit(inputs, returnbeforepool=True)

                tem_s = torch.nn.Softmax(-1)(total/0.1)
                prompt_pred.append(total[:,:init_cls+taskid*cls_inc].argmax(-1).cpu().numpy())
                B = image_features.shape[0]
                for idx, fc in enumerate(all_classifiers[:taskid+1]):
                    task_token = all_tokens[:taskid+1][idx].expand(B, -1, -1)
                    task_token, attn, v = all_tabs[:taskid+1][idx](torch.cat([task_token, prompt, image_features], dim=1), mask_heads=None)
                    task_token = task_token[:, 0]
                    #tem_s = total[:,idx*10:(idx+1)*10]
                    if idx==0:
                        logit = fc(task_token) * tem_s[:,0:init_cls]
                    else:
                        logit = fc(task_token) * tem_s[:,init_cls+(idx-1)*cls_inc:init_cls+idx*cls_inc]
                    #logit = fc(task_token)



                    seperatePreds.append(logit)

            outputs = []
            pred = torch.cat(seperatePreds,dim=-1)
            outputs = pred.argmax(-1).T
            y_pred.append(outputs.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        prompt_pred = np.concatenate(prompt_pred)
        # print(y_pred.T == y_true)
        if args.dil:
            accs.append(np.around((y_pred.T%args.max_cls == y_true%args.max_cls).sum() * 100 / len(y_true), decimals=2))
        else:
            accs.append(np.around((y_pred.T == y_true).sum() * 100 / len(y_true), decimals=2))
        p_accs.append(np.around((prompt_pred.T//args.init_cls == y_true//args.init_cls).sum() * 100 / len(y_true), decimals=2))

        tempacc = []
        p_tempacc = []
        for class_id in range(0, np.max(y_true), _total_classes-_known_classes):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + _total_classes-_known_classes))[0]
            tempacc.append(np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=3))
            p_tempacc.append(np.around((prompt_pred[idxes]//args.init_cls == y_true[idxes]//args.init_cls).sum() * 100 / len(idxes), decimals=3))

        accuracy_table.append(tempacc)
        p_table.append(p_tempacc)

        _known_classes = _total_classes


    np_acctable = np.zeros([taskid + 1, taskid + 1])
    for idxx, line in enumerate(accuracy_table):
        idxy = len(line)
        np_acctable[idxx, :idxy] = np.array(line)
    np_acctable = np_acctable.T

    p_acctable = np.zeros([taskid + 1, taskid + 1])
    for idxx, line in enumerate(p_table):
        idxy = len(line)
        p_acctable[idxx, :idxy] = np.array(line)
    ## import pdb;pdb.set_trace()
    p_acctable = p_acctable.T
    print("Accuracy table:")
    print(np_acctable)
    print("Accuracy curve:")
    print(accs)
    print("FAA: {}".format(accs[-1]))
    forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, taskid])[:taskid])
    print("FF: {}".format(forgetting))
    print("prompt acc:")
    print(p_acctable)
    print(p_accs)

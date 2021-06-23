import torch


def trainer(args, model, train_loader, dev_loader):

    for epoch in args.epochs:
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        total_num = 0
        for i, batch in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels = batch
            batch_num = len(input_ids)
            if args.use_cuda:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                labels = labels.cuda()

            if args.model_type == 'entity_linking':
                logits = model(input_ids, attention_mask, token_type_ids)
                train_loss = model.criterion(logits, labels.float())
                preds = (logits > 0).int()
                train_acc = (preds == labels).float().mean()
            else:
                outputs = model(input_ids, attention_mask, token_type_ids)
                train_loss = model.criterion(outputs, labels)
                _, preds = torch.max(outputs, dim=1)
                train_acc = (preds == labels).float().mean()

            model.optimizer.zero_grad()
            train_loss.backforward()
            model.optimizer.step()

            total_train_loss += train_loss
            total_train_acc += train_acc * batch_num
            total_num += batch_num
            print('batch_%d, train_loss: %d, batch_train_acc: %d', (i, train_loss, train_acc))
        print('epoch_%d, train_loss: %d, epoch_train_acc: %d', (epoch, total_train_loss / total_num, total_train_acc / total_num))


        # eval
        model.eval()
        total_dev_loss = 0
        total_dev_acc = 0
        total_num = 0
        for i, batch in enumerate(dev_loader):
            input_ids, attention_mask, token_type_ids, labels = batch
            batch_num = len(input_ids)
            if args.use_cuda:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                labels = labels.cuda()

            with torch.no_gard():
                if args.model_type == 'entity_linking':
                    logits = model(input_ids, attention_mask, token_type_ids)
                    dev_loss = model.criterion(logits, labels.float())
                    preds = (logits > 0).int()
                    dev_acc = (preds == labels).float().mean()
                else:
                    outputs = model(input_ids, attention_mask, token_type_ids)
                    dev_loss = model.criterion(outputs, labels)
                    _, preds = torch.max(outputs, dim=1)
                    dev_acc = (preds == labels).float().mean()

                total_dev_loss += dev_loss
                total_dev_acc += dev_acc * batch_num
                total_num += batch_num
                print('batch_%d, dev_loss: %d, batch_dev_acc: %d', (i, dev_loss, dev_acc))
        print('epoch_%d, dev_loss: %d, epoch_dev_acc: %d', (epoch, total_dev_loss / total_num, total_dev_acc / total_num))

        torch.save(model.state_dict(), args.save_et_path + f'el_epoch_{epoch}.pth')


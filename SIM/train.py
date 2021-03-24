import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Model
from video_dataset import Dataset
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
import time
import random
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def CLSLOSS(logits, seq_len, batch_size, labels, device):
    lab = F.softmax(labels,dim = 1)
    clsloss = -torch.mean(torch.sum(Variable(lab) * F.log_softmax(logits, dim=1), dim=1), dim=0)
    return clsloss

def InPOINTSLOSS(logits,batch_size,point_idx,gtlabels,device,itr,len):
    # gtlabels = gtlabels / (torch.sum(gtlabels, dim=1, keepdim=True) + 1e-10)
    lab = torch.zeros(0).to(device)
    # print(lab)
    instance_logits = torch.zeros(0).to(device)
    # print(instance_logits)
    for i in range(batch_size):
        labels = torch.from_numpy(np.array([gtlabels[i]])).float().to(device)
        # print(labels.shape)
        # labels = labels / (torch.sum(labels, dim=0, keepdim=True) + 1e-10)
        for k,pt in enumerate(point_idx[i]):
            length = random.randint(0,len)
            # length = 0
            if pt - length < 0:
                start,end = 0,length*2 + 1
            elif pt + length +1 > logits[i].shape[0]:
                strart,end = pt - length*2,pt+1
            else:
                start,end = pt - length,pt + length +1
            tmp_logits = torch.zeros(0).to(device)
            for se in range(start,end):
                tmp_logits = torch.cat([tmp_logits,logits[i][[se]]],dim=0)
            instance_logits = torch.cat([instance_logits,torch.mean(tmp_logits, 0, keepdim=True)],dim=0)
            # instance_logits = torch.cat([instance_logits,logits[i][[pt]]],dim=0)
            lab = torch.cat([lab, labels[0][[k]]], dim=0)
    
    Inploss = -torch.mean(torch.sum(Variable(lab) * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    
    return Inploss



def getNewCAS(cas,action):
    # cas = cas / (abs(max(cas)) + abs(min(cas)))
    # action = action / (abs(max(action) + abs(min(cas))))
    mean1 = np.mean(cas)
    std1 = np.std(cas)
    # print(cas)
    # print(mean1,std1)
    # print(max(cas),min(cas))
    cas = (cas-mean1)/std1
    # print(max(cas),min(cas))
    # print(max(action),min(action))
    mean2 = np.mean(action)
    std2 = np.std(action)
    action = (action-mean2)/std2

    return (cas + action)

def NormalCAS(cas):
    mean1 = np.mean(cas)
    std1 = np.std(cas)
    cas = (cas-mean1)/std1
    return cas

def BACKGROUNDLOSS(logits,action,seq_len,batch_size,point_idx,gtlabels,device):
    # if itr >1500:
    # 加入背景类，    得分低于10%
    # 背景类的选取，先把存在动作类别的CAS求和，然后阈值化，低于10%即认为是背景
    neg_lab = torch.zeros([1,21],dtype=torch.float).to(device)
    neg_lab[0,20] = 1
    weak_lab = torch.zeros(0).to(device)
    target = torch.ones([1,1],dtype=torch.float).to(device)
    # print(neg_lab)
    neg_label = torch.zeros(0).to(device)
    neg_instance_logits = torch.zeros(0).to(device)
    neg_action_instance_logits = torch.zeros(0).to(device)
    k = np.ceil(seq_len/8).astype('int32')
    for i in range(batch_size):
    # 确定每一个视频中出现的动作类别
        # GT = {}
        # labels = torch.from_numpy(np.array([gtlabels[i]])).float().to(device)

        tmplogits = logits[i][:,-1][:seq_len[i]]
        # print(tmplogits.shape)
        tmpaction = -action[i][:,0][:seq_len[i]]
        # print(tmplogits)
        # print(tmpaction)
        # input()
        # print(tmpaction.shape)
        # input()
        # print(tmplogits.shape)
        tmp = getNewCAS(tmplogits,tmpaction)
        gtIndex = point_idx[i]
        # print(gtIndex)
        # print(logits[i][:,-1][:seq_len[i]].shape)
        # print(tmp.shape)
        # print(torch.Tensor(tmp).shape)
        # input()
        _, index = torch.topk(tmp, k=int(k[i]), dim=0,largest=True)       # 取K个background
        # print(index.shape)
        # print(index)
        # input()
        tmp_logits = torch.zeros(0).to(device)
        tmp_action_logits = torch.zeros(0).to(device)
        FLAG = False
        # print(logits.shape)
        # print(index)
        for idx in index:
            # print(idx)
            # print(gtIndex)
            if idx not in gtIndex:
                FLAG = True
                # print(logits[i][[idx]].reshape(1,-1).shape)
                # input()
                tmp_logits = torch.cat([tmp_logits,logits[i][[idx]].reshape(1,-1)],dim=0)
                tmp_action_logits = torch.cat([tmp_action_logits,action[i][[idx]].reshape(1,-1)],dim=0)
        if FLAG:
            # print(tmp_logits.shape)
            neg_label = torch.cat([neg_label,neg_lab],dim = 0)
            neg_instance_logits = torch.cat([neg_instance_logits, torch.mean(tmp_logits, 0, keepdim=True)], dim=0)     #取平均

            weak_lab = torch.cat([weak_lab,target],dim=0)
            neg_action_instance_logits = torch.cat([neg_action_instance_logits, torch.mean(tmp_action_logits, 0, keepdim=True)], dim=0)     #取平均

    neg_SegLoss = -0.1*torch.mean(torch.sum(Variable(neg_label) * F.log_softmax(neg_instance_logits, dim=1), dim=1), dim=0)
    neg_ActLoss =  0.1 * F.binary_cross_entropy(1-torch.sigmoid(neg_action_instance_logits),weak_lab)
    # print(neg_SegLoss)
    # print(neg_ActLoss)
    # input()
    return neg_SegLoss+neg_ActLoss

def sim(f1,f2):
    f1 = torch.transpose(f1,1,0)
    f2 = torch.transpose(f2,1,0)
    sim_loss = 1 - torch.sum(f1 * f2, dim=0) / (torch.norm(f1, 2, dim=0) * torch.norm(f2, 2, dim=0))
    return np.around(sim_loss.data.cpu().numpy(),decimals=4)

def getS1(elem):
    # print(elem[0])
    return elem[0]
def sigmoid(x,situ=True):
    if situ:
        return  1/(1+np.exp(-x))
    else:
        return 1
def Reverse(x):
    return (np.exp(-x))

def spliteCAS(CAS,interval,point_idx,actionNoExitIndex=[]):
    # actionNoExitIndex = []
    if len(point_idx) > 1:      # 存在多个点，可能会相交
        for j in range(1,len(point_idx)):
            Exist = [(t,ex) for t,ex in enumerate(interval) if (point_idx[j-1] in ex and point_idx[j] in ex)]
            if Exist:
                idxtmp = Exist[0][0];Exist = Exist[0][1]
                if point_idx[j] - point_idx[j-1] >1:
                    minVal = min(CAS[point_idx[j-1]+1:point_idx[j]])
                    # minIdx = []
                    # if minVal < threshold:
                    # minIdx = [(t + point_idx[j-1]+1) for t,m in enumerate(CAS[point_idx[j-1]+1:point_idx[j]]) if m < threshold]
                    # else:
                    minIdx = [t + point_idx[j-1]+1 for t,m in enumerate(CAS[point_idx[j-1]+1:point_idx[j]]) if m == minVal]
                    # print('minVal:{},minIdx:{}'.format(minVal,minIdx))

                    actionNoExitIndex.append([idx for idx in minIdx])

                    interval[idxtmp] = []
                    AD = 0
                    if Exist[0] - AD > 0:
                        interval.append([v for v in range(Exist[0]-AD,minIdx[0])])
                    else:
                        interval.append([v for v in range(Exist[0],minIdx[0])])
                    # print([v for v in range(Exist[0],minIdx)])
                    interval.append([v for v in range(minIdx[-1]+1,Exist[-1]+1)])
    interval = [Int for Int in interval if Int]         # 移除[],即移除元素为NULL的list
    return interval,actionNoExitIndex

def UnionSegment(interval,itl1,itl2,pt):
    tmp = list(set(itl1).union(set(itl2)))
    # 此处可能会出现这样一种情况：tmp与前后segment有重合！
    tmp.sort()
    # tmp = [itl for itl in spliteCAS(CAS,[tmp],point_idx)[0] if pt in itl][0]
    idx = [idx for idx,itl in enumerate(interval) if pt in itl][0]
    # 左侧：
    if idx > 0:
        if list(set(interval[idx-1]).intersection(set(tmp))):
            tmp = [it for it in tmp if it > interval[idx-1][-1]+1]
        # elif 
    if idx+1 < len(interval):
        if list(set(interval[idx+1]).intersection(set(tmp))):
            tmp = [it for it in tmp if it < interval[idx+1][0]-1]
    tmp.sort()
    return tmp

def interPredSegment(interval,actITL,point_idx):
    result = [itl for itl in interval]
    for pt in point_idx:
        Itl = [idx for idx,itl in enumerate(result) if pt in itl]
        Atl = [idx for idx,itl in enumerate(actITL) if pt in itl]
        if Itl and Atl:
            result[Itl[0]] = list(set(interval[Itl[0]]).intersection(set(actITL[Atl[0]])))
            result[Itl[0]].sort()
    return result

def UnionPreSegment(interval,actITL,point_idx):
    result = [itl for itl in interval]
    for pt in point_idx:
        Itl = [idx for idx,itl in enumerate(interval) if pt in itl]
        Atl = [idx for idx,itl in enumerate(actITL) if pt in itl]
        if Itl and Atl:
            # interval[Itl[0]] = list(set(interval[Itl[0]]).union(set(actITL[Atl[0]])))
            result[Itl[0]] = UnionSegment(result,interval[Itl[0]],actITL[Atl[0]],pt)
            result[Itl[0]].sort()
    return interval

def deleteWeakSegment(sigment,feature,CAS,Action,interval,point_idx,rate=1.0):
    """
    把距离含标注点较近的阈值片段移除出去，在后期不加以处理。即：这部分既不作为正例，也不作为负例。
    """
    # coarseSegment = []
    # threshold = min(CAS) + 0.1*(max(CAS) - min(CAS))
    # print(interval)
    # cas = [sigmoid(float(c)) for c in CAS]
    cas = getNewCAS(CAS,Action)
    # cas = [sigmoid(float(c)) for c in cas]
    # cas = [1 for k in CAS]
    for pt in point_idx:
        interval.sort()
        # print(interval)
        sig = [s for s in sigment if s[0]<=pt and s[-1]>=pt][0]
        # 阈值划分CAS+Action:
        threshold = min(CAS) + (CAS[pt]-min(CAS)) / 2.0
        cas_pred = np.concatenate([np.zeros(1),(CAS>=threshold).astype('float32'),np.zeros(1)], axis=0)
        cas_pred_diff = [cas_pred[idt]-cas_pred[idt-1] for idt in range(1,len(cas_pred))]
        s = [idk for idk,item in enumerate(cas_pred_diff) if item==1]
        e = [idk for idk,item in enumerate(cas_pred_diff) if item==-1]
        # print(s,'\n',e)
        itl2CAS = [seg for seg in [[k for k in range(si,ei)] for si,ei in zip(s,e)] if pt in seg]        # 每个segment的区间,si不包括
        itl2CAS = [itl for itl in spliteCAS(CAS,itl2CAS,point_idx)[0] if pt in itl][0]
        threshold = min(Action) + (Action[pt] - min(Action)) / 2.0
        action_pred = np.concatenate([np.zeros(1),(Action>=threshold).astype('float32'),np.zeros(1)], axis=0)
        action_pred_diff = [action_pred[idt]-action_pred[idt-1] for idt in range(1,len(action_pred))]
        s = [idk for idk,item in enumerate(action_pred_diff) if item==1]
        e = [idk for idk,item in enumerate(action_pred_diff) if item==-1]
        # print(s,'\n',e)
        itl2Action = [seg for seg in [[k for k in range(si,ei)] for si,ei in zip(s,e)] if pt in seg]        # 每个segment的区间,si不包括
        itl2Action = [itl for itl in spliteCAS(Action,itl2Action,point_idx)[0] if pt in itl][0]
        # 取并集：CAS+Action标注点的预测
        # CasUnionAction = list(set(itl2CAS).union(set(itl2Action)))
        CasUnionAction = list(set(itl2CAS).intersection(set(itl2Action)))
        CasUnionAction.sort()

        f1 = feature[[pt]]
        ITL = [(idx,itl) for idx,itl in enumerate(interval) if pt in itl and len(itl)>1]
        threshold = np.min(CAS) + (np.max(CAS)-np.min(CAS))*0.1
        originThreshold = np.min(CAS) + (np.max(CAS)-np.min(CAS))*0.5
        if ITL:
            Index = ITL[0][0]
            origin = ITL[0][1]
            # print
            Length = int(rate * len(ITL))
            # if Length < 5:
            #     Length = 5
            tmpITL = spliteWeakSegment(getNewCAS(CAS,Action),origin,pt)
            if len(tmpITL) > 1:
                ITL = tmpITL
            else:
                ITL = origin
            # 不需要考虑越界的情况：小于0或者大于最大长度，因为取待选片段取交集。
            Left = [k for k in range(ITL[0],ITL[0]-Length) if k >-1]
            Right = [k for k in range(ITL[-1],ITL[-1]+Length) if k < len(ITL)]
            LR = list(set(Left).union(set(Right)))
            CasUnionAction = list(set(LR).union(set(CasUnionAction)))

            score = []
            tmpS = []

            idxLeft = pt
            idxRight = pt

            for idx in ITL:
                f2 = feature[[idx]]
                tmpSim = sim(f1,f2)[0]
                tmpS.append(cas[idx]*Reverse(tmpSim))
                if idx != pt:
                    score.append(cas[idx]*Reverse(tmpSim))
            # print('MaxS:{:.4f},MinS:{:.4f},MeanS:{:.4f}'.format(max(score),min(score),np.mean(score)))
            # print(score)
            # thre = np.mean(score)
            # thre = np.min(score)
            if len(ITL) > 2:
                thre = np.min(score) + (np.max(score)-np.min(score))*0.5
            else:
                thre = np.min(tmpS) + (np.max(tmpS)-np.min(tmpS))*0.5
            for k in range(len(ITL)):
                if tmpS[k] >= thre:
                    idxLeft = ITL[k]
                    break
            for k in range(len(ITL)-1,-1,-1):
                if tmpS[k] >= thre:
                    idxRight = ITL[k]
                    break

            
            # if origin != ITL:
            #     interval[Index] = ITL
            #     maybe = list(set(origin).difference(set(ITL)))
            #     maybe.sort()
            #     L = [m for m in maybe if m <pt-1 and CAS[m] > originThreshold]
            #     if len(L)>1:
            #         interval.append(L)
            #     R = [m for m in maybe if m >pt+1 and CAS[m] > originThreshold]
            #     if len(R)>1:
            #         interval.append(R)
            #     interval.sort()
            # Index = [idx for idx,itl in enumerate(interval) if pt in itl][0]
            # CasUnionAction.sort()
            # BG = []
            # CUA,BG = spliteCAS(CAS,[CasUnionAction],point_idx,BG)
            # # print(BG)
            # CUA = [itl for itl in CUA if pt in itl][0]
            # idxdel = []
            # # print(pt,ITL[0][0])
            # # 标注点左侧候选点：
            # # 要求：1.不在候选片段中 2.在候选片段中，但是此候选片段没有其他标注点
            # tmpPoint = [idx for idx in CUA if idx < ITL[0]]
            # LeftPoint = []
            # for lp in tmpPoint:
            #     tmpI = [itl for itl in interval if lp in itl]
            #     if not tmpI:
            #         # 不在候选片段中
            #         LeftPoint.append(lp)
            #     elif not [pit for pit in point_idx if pit in tmpI]:
            #         # 虽然在候选片段中，但是候选片段中没有标注点
            #         LeftPoint.extend(tmpI[0])
            #     else:
            #         break
            
            # LeftPoint = list(set(LeftPoint))
            # LeftPoint.sort()
            # for dx in LeftPoint:
            #     f2 = feature[[dx]]
            #     tmpSim = sim(f1,f2)[0]
            #     score.append(cas[dx]*Reverse(tmpSim))
            #     if cas[dx]*Reverse(tmpSim) >= thre:
            #         if np.min(cas[dx:ITL[0]]) >= 0.0:
            #             if idxLeft > dx:
            #                 idx = [idx for idx,itl in enumerate(interval) if dx in itl]
            #                 if idx:
            #                     idxdel.extend([t for t in range(idx[0],Index)])
            #                 idxLeft = dx
            #                 break
            #         else:
            #             break

            # # 标注点右侧
            # tmpPoint = [idx for idx in CUA if idx > ITL[-1]]
            # RightPoint = []
            # for lp in tmpPoint:
            #     tmpI = [itl for itl in interval if lp in itl]
            #     if not tmpI:
            #         # 不在候选片段中
            #         RightPoint.append(lp)
            #     elif not [pit for pit in point_idx if pit in tmpI]:
            #         # 虽然在候选片段中，但是候选片段中没有标注点
            #         RightPoint.extend(tmpI[0])
            #     else:
            #         break
            # RightPoint = list(set(RightPoint))
            # RightPoint.sort(reverse=True)
            # for dx in RightPoint:
            #     f2 = feature[[dx]]
            #     tmpSim = sim(f1,f2)[0]
            #     score.append(cas[dx]*Reverse(tmpSim))
            #     if cas[dx]*Reverse(tmpSim) >= thre:
            #         if np.min(cas[ITL[-1]+1:dx+1]) >= 0.0:
            #             if idxRight < dx:
            #                 idx = [idx for idx,itl in enumerate(interval) if dx in itl]
            #                 if idx:
            #                     idxdel.extend([t for t in range(Index+1,idx[0]+1)])
            #                 idxRight = dx
            #                 break
            #         else:
            #             break
            if idxRight-idxLeft<3:
                idxLeft = ITL[0]
                idxRight = ITL[-1]
            interval[Index] = [d for d in range(idxLeft,idxRight+1)]
            score = []
            for idx in interval[Index]:
                f2 = feature[[idx]]
                tmpSim = sim(f1,f2)[0]
                # tmpdict = dict()
                # tmpdict[idx] = tmpSim
                # other.append(tmpdict)
                score.append(cas[idx]*Reverse(tmpSim))
            # print(CAS[interval[Index][0]:interval[Index][-1]+1])
            # print(interval[Index])
            # print(len(CAS))
            segScore = []
            for idx in range(sig[0],sig[-1]+1):
                f2 = feature[[idx]]
                tmpSim = sim(f1,f2)[0]
                # tmpdict = dict()
                # tmpdict[idx] = tmpSim
                # other.append(tmpdict)
                segScore.append(cas[idx]*Reverse(tmpSim))
            utils.write_seg('TP0206_mid_20200713',cas[interval[Index][0]:interval[Index][-1]+1],cas[sig[0]:sig[-1]+1],pt,interval[Index],sig,score,thre,ITL,segScore)
            
            idxdel = list(set(idxdel))
            idxdel = [interval[t] for t in idxdel]
            for i in idxdel:
                interval.remove(i)
            # print(interval)
            # idx = [idx for idx,itl in enumerate(interval) if pt in itl][0]
            # # print('idx:',interval[idx])
            # interval[idx] = spliteWeakSegment(feature,cas,interval[idx],pt,thre)

        else:
            for itl in interval:
                if not [idx for idx in point_idx if idx in itl]:
                    if list(set(itl).intersection(set(CasUnionAction))):
                        interval.remove(itl)
    # idxdel = list(set(idxdel))
    # for i in idxdel:
    #     interval.remove(interval[i])
    return interval

def spliteWeakSegment(logits,predSegment,point_idx,rate = 0.6):
    """
    对单独阈值片段进行处理，如果，出现波谷值小于maxScore的50%，则进行切分：
    怎么确定波谷呢？
    阈值划分：
    0.5开始，0.05为步长如果存在超过一个segment，则进行划分！存在point的作为正例，不存在的，移除出去。
    """
    tmp = logits[predSegment[0]:predSegment[-1]+1]          # 对应的预测分数
    # act = action[predSegment[0]:predSegment[-1]+1]
    # tmp = getNewCAS(tmp,act)
    # print(len(predSegment),'\t',len(tmp))       # 12       12
    # print(predSegment)
    # print(tmp)
    idx = point_idx - predSegment[0]                        # 第几个？
    # print(point_idx,'\t',idx)
    add = predSegment[0]
    difference = max(tmp)-min(tmp)
    minV = min(tmp)
    pRate = (tmp[idx] - min(tmp)) / difference
    if pRate < rate:
        rate = pRate
    tmpRate = [0.01]       # Rate = [0.01,0.06,0.11,0.16,0.21,0.26,0.31,0.36,0.41,0.46,0.51,0.56]
    if rate < tmpRate[0]:
        tmpRate[0] = 0.0
    r = tmpRate[0]
    while (r+0.1) < rate:
        r += 0.1
        tmpRate.append(r)
    # print(tmpRate)
    # 当预测值出现较大波动时(diff>0.05)，拆分预测片段。
    for tR in tmpRate:
        threshold = minV + difference*tR
        vid_pred = np.concatenate([np.zeros(1),(tmp>=threshold).astype('float32'),np.zeros(1)], axis=0)
        vid_pred_diff = [vid_pred[idt]-vid_pred[idt-1] for idt in range(1,len(vid_pred))]
        s = [idk for idk,item in enumerate(vid_pred_diff) if item==1]
        e = [idk for idk,item in enumerate(vid_pred_diff) if item==-1]
        # print(s,'\n',e)
        interval = [[k for k in range(si,ei)] for si,ei in zip(s,e)]        # 每个segment的区间,si不包括
        if not interval:
            interval.append([idx])
        # print(interval,idx)
        interval[0] = [t for t in range(interval[0][-1]+1)]                     # 处理首部
        interval[-1] = [t for t in range(interval[-1][0],interval[-1][-1]+1)]   # 处理尾部

        # print(idx)
        # print(interval)
        index = [t for t,itl in enumerate(interval) if idx in itl][0]
        # print(index)
        if len(interval) > 1:
            # 标注片段左侧：
            for t in range(index-1,-1,-1):  # index=1 t=0
                MinV = min(tmp[interval[t][-1]+1:interval[index][0]])
                MaxV = max(tmp[interval[t][0]:interval[t][-1]+1])
                if (MaxV - MinV) <= 0.1 * difference:
                    interval[index] = [t for t in range(interval[t][0],interval[index][-1]+1)]
                else:
                    break
            # 标注点右侧：
            for t in range(index+1,len(interval)):
                MinV = min(tmp[interval[index][-1]+1:interval[t][0]])
                MaxV = max(tmp[interval[t][0]:interval[t][-1]+1])
                if (MaxV - MinV) <= 0.1 * difference:
                    interval[index] = [t for t in range(interval[index][0],interval[t][-1]+1)]
                else:
                    break
        # tmpPredSegment = interval[index]
        # print(interval[index])
        # input()
        tmp = tmp[interval[index][0]:interval[index][-1]+1]           # 修正区间
        idx -= interval[index][0]
        add += interval[index][0]
        tmpSegment = [k for k in range(0,len(interval[index]))]

        # input()
    # tmpGtSegment = [itl for itl in interval if idx in itl]
    # print(tmpSegment)
    gtSegment = [ps+add for ps in tmpSegment]
    # print(gtSegment)
    return gtSegment

# def interPredSegment(interval,actITL,point_idx):
#     for pt in point_idx:
#         Itl = [idx for idx,itl in enumerate(interval) if pt in itl]
#         Atl = [idx for idx,itl in enumerate(actITL) if pt in itl]
#         if Itl and Atl:
#             interval[Itl[0]] = list(set(interval[Itl[0]]).intersection(set(actITL[Atl[0]])))
#             interval[Itl[0]].sort()
#     return interval

def SEGMENT(sigment,feature,seq_len,logits,action,batch_size,point_idx,gtlabels, args,device):
    '''
    gtlabel:list[array(),....]
    '''
    actionExit = []
    actionNoExit = []
    for i in range(batch_size):
        # tmplogits = logits[i][:][:seq_len[i]]
        # print(tmplogits.shape)
        # tmpaction = -action[i][:,0][:seq_len[i]]
        # 确定每一个视频中出现的动作类别
        GT = {}
        actionExitIndex = []
        actionNoExitIndex = []
        # labels = torch.from_numpy(np.array([gtlabels[i]])).float().to(device)
        for idx,gtL in zip(point_idx[i],gtlabels[i]):
            # print(idx,gtL)      # 276 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            c = int(np.where(gtL==1)[0])
            if c in GT.keys():
                temp = GT[c]
                temp.append(idx)
                GT[c] = temp
            else:
                GT[c] = [idx]

        # 上述过程，求出了类别c对应的可能区段
        for c in GT.keys():     # 动作存在
            # print(logits[i][:,c].shape)
            tmp = logits[i][:,c][:seq_len[i]].data.cpu().numpy().reshape(logits[i][:,c][:seq_len[i]].data.cpu().numpy().shape[0],)
            # print(tmp.shape)    # (818,)
            # print(logits[i][[c]].shape)       # torch.Size([1, 20])
            threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp))*0.5  if not args.activity_net else 0  # 阈值
            # print(tmp>threshold)
            vid_pred = np.concatenate([np.zeros(1),(tmp>threshold).astype('float32'),np.zeros(1)], axis=0)
            vid_pred_diff = [vid_pred[idt]-vid_pred[idt-1] for idt in range(1,len(vid_pred))]
            s = [idk for idk,item in enumerate(vid_pred_diff) if item==1]
            e = [idk for idk,item in enumerate(vid_pred_diff) if item==-1]
            # print(s,'\n',e)
            interval = [[k for k in range(si,ei)] for si,ei in zip(s,e) if (ei - si)>1]        # 每个segment的区间,si不包括
            # [0,1,1,0]->[1,0,-1]->(0,2)->[[0,1],]        # 不要想太多，在前面已经把首尾补零了。

            GT[c] = list(set(GT[c]))         # 升序
            # print(tmp.sort())
            GT[c].sort()


            # 处理区段，把存在多点的进行拆分
            interval,actionNoExitIndex = spliteCAS(tmp,interval,GT[c],actionNoExitIndex)


            act = action[i][:seq_len[i]].data.cpu().numpy().reshape(action[i][:seq_len[i]].data.cpu().numpy().shape[0],)
            threshold = min(act) + (max(act) - min(act)) / 2.0
            action_pred = np.concatenate([np.zeros(1),(act>=threshold).astype('float32'),np.zeros(1)], axis=0)
            action_pred_diff = [action_pred[idt]-action_pred[idt-1] for idt in range(1,len(action_pred))]
            s = [idk for idk,item in enumerate(action_pred_diff) if item==1]
            e = [idk for idk,item in enumerate(action_pred_diff) if item==-1]
            # print(s,'\n',e)
            # itl2Action = [seg for seg in [[k for k in range(si,ei)] for si,ei in zip(s,e)] if pt in seg][0]        # 每个segment的区间,si不包括
            actITL = [[k for k in range(si,ei)] for si,ei in zip(s,e) if (ei - si)>1]
            actITL,_ = spliteCAS(act,actITL,GT[c])
            interval = UnionPreSegment(interval,actITL,GT[c])
            interval.sort()
            # interval = SimPredSegment(feature[i],tmp,interval,actITL,GT[c])# print(act.shape)
            # interval = interPredSegment(interval,actITL,GT[c])
            # print(interval)
            # interval = deleteWeakSegment(feature[i],tmp,act,interval,GT[c],rate=1)
            # interval = interPredSegment(interval,actITL,GT[c])# print(act.shape)
            interval = deleteWeakSegment(sigment[i],feature[i],tmp,act,interval,GT[c],rate=1)
            # tmp = getNewCAS(tmp,act)
            # 构造正例，即存在point的区段
            index_exit = []     # 保存存在point的区段的index
            for idx in GT[c]:
                ind = int([i for i,p in enumerate(point_idx[i]) if p == idx][0])
                # lab = torch.cat([lab, labels[0][[ind]]], dim=0)
                # tmp_logits = torch.zeros(0).to(device)
                ITL = [itl for itl in interval if idx in itl]       # ITL 是idx存在的区段，有可能为NULL
                if ITL:
                    # posSegment = spliteWeakSegment(feature,tmp,ITL[0],idx)
                    posSegment = ITL[0]
                    # print(posSegment)
                    # input()
                    # actionExitIndex.append([idx for idx in ITL[0]])
                    actionExitIndex.append([tmpIdx for tmpIdx in posSegment])
                    interval.remove(ITL[0])
                    # index_exit.append([t for t,itl in enumerate(interval) if idx in itl])
                    # for Idx in ITL[0]:
                    #     actionExitIndex.append(Idx)
                else:
                    # length = random.randint(0,2)
                    length = 0
                    if idx - length < 0:
                        start,end = 0,length*2+1
                    elif idx + length +1 > logits[i].shape[0]:
                        strart,end = logits[i].shape[0] - length*2 -1,logits[i].shape[0]
                    else:
                        start,end = idx - length,idx + length + 1
                    actionExitIndex.append([idx for idx in range(start,end)])
                    # for se in range(start,end):
                    #     actionExitIndex.append(se)

            # 构造负例，即得分很高，但是不存在point的区段
            for ITL in interval:
                actionNoExitIndex.append([idx for idx in ITL])
        actionExit.append(actionExitIndex)
        actionNoExit.append(actionNoExitIndex)
    return (actionExit,actionNoExit)


def SEGMENTLOSS(logits,batch_size,point_idx,gtlabels,actionExit,actionNoExit,device):
    '''
    gtlabel:list[array(),....]
    '''
    lab = torch.zeros(0).to(device)
    neg_lab = torch.zeros([1,21],dtype=torch.float).to(device)
    neg_lab[0,20] = 1
    weak_neg_label = torch.zeros(0).to(device)
    instance_logits = torch.zeros(0).to(device)
    weak_neg_instance_logits = torch.zeros(0).to(device)
    for i in range(batch_size):
        # 确定每一个视频中出现的动作类别
        GT = {}
        labels = torch.from_numpy(np.array([gtlabels[i]])).float().to(device)
        for idx,gtL in zip(point_idx[i],gtlabels[i]):
            # print(idx,gtL)      # 276 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            c = int(np.where(gtL==1)[0])
            if c in GT.keys():
                temp = GT[c]
                temp.append(idx)
                GT[c] = temp
            else:
                GT[c] = [idx]

        # 上述过程，求出了类别c对应的可能区段
        for c in GT.keys():     # 动作存在
            # 构造正例，即存在point的区段
            for idx in GT[c]:
                ind = int([i for i,p in enumerate(point_idx[i]) if p == idx][0])
                lab = torch.cat([lab, labels[0][[ind]]], dim=0)
                # print('idx:{}'.format(idx))
                tmp_logits = torch.zeros(0).to(device)
                ITL = [itl for itl in actionExit[i] if idx in itl][0]       # ITL 是idx存在的区段
                for Idx in ITL:
                   tmp_logits = torch.cat([tmp_logits,logits[i][[Idx]]],dim=0)
                instance_logits = torch.cat([instance_logits, torch.mean(tmp_logits, 0, keepdim=True)], dim=0)     #取平均

        # 构造负例，即得分很高，但是不存在point的区段
        for ITL in actionNoExit[i]:
            tmp_logits = torch.zeros(0).to(device)
            FLAG = False
            for Idx in ITL:
                FLAG = True
                tmp_logits = torch.cat([tmp_logits,logits[i][[Idx]]],dim=0)
            if FLAG:
                weak_neg_label = torch.cat([weak_neg_label,neg_lab],dim = 0)
                weak_neg_instance_logits = torch.cat([weak_neg_instance_logits, torch.mean(tmp_logits, 0, keepdim=True)], dim=0)     #取平均
    weak_neg_SegLoss =  -torch.mean(torch.sum(Variable(weak_neg_label) * F.log_softmax(weak_neg_instance_logits , dim=1), dim=1), dim=0)
    SegLoss = -torch.mean(torch.sum(Variable(lab) * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    # penalize = torch.mean(torch.sum(F.softmax(instance_logits, dim=1), dim=1), dim=0)
    # SegLoss += 0.001 * penalize
    return SegLoss + 0.1 * weak_neg_SegLoss

def UnionActionNoExit(ANE_F,ANE_R,batch_size):
    actionNoExit = []
    for i in range(batch_size):
        TMP = []
        for ane_f in ANE_F[i]:
            # print(ane_f)
            # input()
            t_max = ane_f[-1]
            t_min = ane_f[0]
            # FLAG = True
            for ane_r in ANE_R[i]:
                if list(set(ane_f).intersection(set(ane_r))):        # 交集不为空
                    # FLAG = False
                    ANE_R[i].remove(ane_r)
                    if t_min > ane_r[0]:
                        t_min = ane_r[0]
                    if t_max < ane_r[-1]:
                        t_max = ane_r[-1]
            TMP.append([i for i in range(t_min,t_max+1)])
        for ane_r in ANE_R:
            TMP.append(ane_r)
        actionNoExit.append(TMP)
    return actionNoExit

def InterActionExit(ANE_F,ANE_R,point_indexs,batch_size):
    actionExit = []
    for i in range(batch_size):
        TMP = []
        for idx in point_indexs[i]:
            ITL_f = [itl for itl in ANE_F[i] if idx in itl][0]
            ITL_r = [itl for itl in ANE_R[i] if idx in itl][0]
            # TMP.append(list(set(ITL_f).union(set(ITL_r))))
            TMP.append(list(set(ITL_f).intersection(set(ITL_r))))
        actionExit.append(TMP)
    return actionExit


def ACTIONLOSS(action,seq_len,batch_size,point_idx,device,len):
    lab = torch.zeros(0).to(device)
    instance_logits = torch.zeros(0).to(device)
    target = torch.ones([1,1],dtype=torch.float).to(device)
    for i in range(batch_size):
        flag = False
        tmp_logits = torch.zeros(0).to(device)
        for idx in point_idx[i]:
            flag = True
            length = random.randint(0,len)
            # length = 0
            if idx - length < 0:
                start,end = 0,length*2+1
            elif idx + length +1 > action[i].shape[0]:
                strart,end = action[i].shape[0] - length*2 -1,action[i].shape[0]
            else:
                start,end = idx - length,idx + length + 1
            for se in range(start,end):
                tmp_logits = torch.cat([tmp_logits,action[i][[se]]],dim=0)
        if flag:
            lab = torch.cat([lab,target],dim=0)
            instance_logits = torch.cat([instance_logits,torch.mean(tmp_logits, 0, keepdim=True)],dim=0)

    actloss = F.binary_cross_entropy_with_logits(instance_logits,lab)
    return actloss

def ACTION2SEGLOSS(action,actionExit,actionNoExit,batch_size,device):
    lab = torch.zeros(0).to(device)
    weak_lab = torch.zeros(0).to(device)
    instance_logits = torch.zeros(0).to(device)
    neg_instance_logits = torch.zeros(0).to(device)
    weak_neg_instance_logits = torch.zeros(0).to(device)
    target = torch.ones([1,1],dtype=torch.float).to(device)
    for i in range(batch_size):

        # 处理负例 top-k
        # _, negIndex = torch.topk(action[i][:seq_len[i]], k=int(k[i]), dim=0,largest=False)       # 取K个background
        # negIndex = [idx for idx in negIndex if (idx not in actionNoExit or idx not in actionExit)]
        # tmp_logits = torch.zeros(0).to(device)
        # for idx in negIndex:
        #     tmp_logits = torch.cat([tmp_logits,action[i][[t]]],dim=0)
        # neg_instance_logits = torch.cat([neg_instance_logits,torch.mean(tmp_logits, 0, keepdim=True)],dim=0)
        # 处理正例
        # FLAG = False
        tmp_logits = torch.zeros(0).to(device)
        for itl in actionExit[i]:
            for idx in itl:
                tmp_logits = torch.cat([tmp_logits,action[i][[idx]]],dim=0)
        lab = torch.cat([lab,target],dim=0)
        instance_logits = torch.cat([instance_logits,torch.mean(tmp_logits, 0, keepdim=True)],dim=0)
        # 处理弱负例
        tmp_logits = torch.zeros(0).to(device)
        FLAG = False
        for itl in actionNoExit[i]:
            for idx in itl:
                FLAG = True
                tmp_logits = torch.cat([tmp_logits,action[i][[idx]]],dim=0)
        if FLAG:
            weak_lab = torch.cat([weak_lab,target],dim=0)
            weak_neg_instance_logits = torch.cat([weak_neg_instance_logits,torch.mean(tmp_logits, 0, keepdim=True)],dim=0)


    actloss = F.binary_cross_entropy_with_logits(instance_logits,lab)
    # actloss +=  0.05 * F.binary_cross_entropy(1-torch.sigmoid(neg_instance_logits),lab)
    # if weak_flag:
    actloss +=  0.1 * F.binary_cross_entropy(1-torch.sigmoid(weak_neg_instance_logits),weak_lab)
    return actloss


def train(itr, dataset, args, model, optimizer, logger, device):

    # Batch fprop
    features, labels,gtlabel, count_labels,point_indexs,segment = dataset.load_data()
    # hyperClsLoss = 1.0
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:,:np.max(seq_len),:]

    features = torch.from_numpy(features).float().to(device)
    feature_f = features[:, :, 1024:]
    feature_r = features[:, :, :1024]
    gtSegment = []
    for i in range(args.batch_size):
        seg = []
        for k in range(len(segment[i])):
            # td = []
            td = [int(segment[i][k][0]*25//16),int(segment[i][k][-1]*25//16)]
            seg.append(td)
        gtSegment.append(seg)
    # point_feature = torch.from_numpy(point_feature).float().to(device)
    # print(point_feature.shape)
    labels = torch.from_numpy(labels).float().to(device)
    count_labels = torch.from_numpy(count_labels).float().to(device)

    logits_f, logits_r, tcam,score_f,score_r,score_all,attention_f,attention_r,attention_all,action_f,action_r,action_all = model(Variable(features), device ,seq_len=torch.from_numpy(seq_len).to(device))

    # print(labels.shape,logits_f.shape,score_f.shape,point_indexs.shape)
    total_loss = 0.0

    
    if itr > 6000:
        if itr %2:
            actionExit_f,actionNoExit_f = SEGMENT(gtSegment,feature_f,seq_len,logits_f, action_f ,args.batch_size,point_indexs,gtlabel,args,device)
            input('over!')
            actionExit_r,actionNoExit_r = SEGMENT(gtSegment,feature_r,seq_len,logits_r,action_r, args.batch_size,point_indexs,gtlabel,args,device)
            actionExit,actionNoExit = SEGMENT(gtSegment,features,tcam,seq_len,action_all, args.batch_size,point_indexs,gtlabel,args,device)

            segloss_f = SEGMENTLOSS(logits_f, args.batch_size,point_indexs,gtlabel,actionExit_f,actionNoExit_f,device)
            segloss_r = SEGMENTLOSS(logits_r, args.batch_size,point_indexs,gtlabel,actionExit_r,actionNoExit_r,device)
            segment_final = SEGMENTLOSS(tcam, args.batch_size,point_indexs,gtlabel,actionExit,actionNoExit,device)
            segloss = segloss_f + segloss_r + segment_final

            actloss_f = ACTION2SEGLOSS(action_f,actionExit_f,actionNoExit_f,args.batch_size,device)
            actloss_r = ACTION2SEGLOSS(action_r,actionExit_r,actionNoExit_r,args.batch_size,device)
            actloss_final = ACTION2SEGLOSS(action_all,actionExit,actionNoExit,args.batch_size,device)
            actloss = actloss_r + actloss_f + actloss_final
            
            total_loss += segloss + actloss
        else:
            iploss_f = InPOINTSLOSS(logits_f, args.batch_size,point_indexs,gtlabel,device,itr,len=0)
            iploss_r = InPOINTSLOSS(logits_r, args.batch_size,point_indexs,gtlabel,device,itr,len=0)
            iploss_final = InPOINTSLOSS(tcam, args.batch_size,point_indexs,gtlabel,device,itr,len=0)
            iploss = iploss_f + iploss_r + iploss_final

            actloss_f = ACTIONLOSS(action_f,seq_len,args.batch_size,point_indexs,device,len=0)
            actloss_r = ACTIONLOSS(action_r,seq_len,args.batch_size,point_indexs,device,len=0)
            actloss_final = ACTIONLOSS(action_all,seq_len,args.batch_size,point_indexs,device,len=0)
            actloss = actloss_r + actloss_f + actloss_final

            total_loss += iploss + actloss
    else:
        if itr %2:
            iploss_f = InPOINTSLOSS(logits_f, args.batch_size,point_indexs,gtlabel,device,itr,len=2)
            iploss_r = InPOINTSLOSS(logits_r, args.batch_size,point_indexs,gtlabel,device,itr,len=2)
            iploss_final = InPOINTSLOSS(tcam, args.batch_size,point_indexs,gtlabel,device,itr,len=2)
            iploss = iploss_f + iploss_r + iploss_final

            actloss_f = ACTIONLOSS(action_f,seq_len,args.batch_size,point_indexs,device,len=2)
            actloss_r = ACTIONLOSS(action_r,seq_len,args.batch_size,point_indexs,device,len=2)
            actloss_final = ACTIONLOSS(action_all,seq_len,args.batch_size,point_indexs,device,len=2)
            actloss = actloss_r + actloss_f + actloss_final
        else:
            iploss_f = InPOINTSLOSS(logits_f, args.batch_size,point_indexs,gtlabel,device,itr,len=0)
            iploss_r = InPOINTSLOSS(logits_r, args.batch_size,point_indexs,gtlabel,device,itr,len=0)
            iploss_final = InPOINTSLOSS(tcam, args.batch_size,point_indexs,gtlabel,device,itr,len=0)
            iploss = iploss_f + iploss_r + iploss_final

            actloss_f = ACTIONLOSS(action_f,seq_len,args.batch_size,point_indexs,device,len=0)
            actloss_r = ACTIONLOSS(action_r,seq_len,args.batch_size,point_indexs,device,len=0)
            actloss_final = ACTIONLOSS(action_all,seq_len,args.batch_size,point_indexs,device,len=0)
            actloss = actloss_r + actloss_f + actloss_final
        total_loss += iploss + actloss


        # else:

    clsloss_f = CLSLOSS(score_f, seq_len, args.batch_size, labels, device)
    clsloss_r = CLSLOSS(score_r, seq_len, args.batch_size, labels, device)
    clsloss_final = CLSLOSS(score_all, seq_len, args.batch_size, labels, device)
    clsloss = clsloss_f + clsloss_r + clsloss_final
        # clsloss = 
        # loss_norm = torch.mean(torch.norm(attention_all, p=1, dim=1))
        # total_loss += 0.0001 * loss_norm
    # clsloss = clsloss_f + clsloss_r + clsloss_final
    total_loss += clsloss

    logger.log_value('total_loss', total_loss, itr)
    print('Iteration: %d, Loss: %.3f' %(itr, total_loss.data.cpu().numpy()))

    optimizer.zero_grad()
    if total_loss > 0:
        total_loss.backward()
    if total_loss > 0:
        optimizer.step()
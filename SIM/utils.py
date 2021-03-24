import numpy as np

def str2ind(categoryname,classlist):
   # print(categoryname,classlist[0].decode('utf-8'))
   return [i for i in range(len(classlist)) if categoryname==classlist[i].decode('utf-8')][0]

def strlist2indlist(strlist, classlist):
	return [str2ind(s,classlist) for s in strlist]

def strlist2multihot(strlist, classlist):
	return np.sum(np.eye(len(classlist))[strlist2indlist(strlist,classlist)], axis=0)

def idx2multihot(id_list,num_class):
   return np.sum(np.eye(num_class)[id_list], axis=0)

def random_extract(feat,point,label, t_max):
   r = np.random.randint(len(feat)-t_max)
   # print(r)
   return (feat[r:r+t_max],[p-r for p in point if (p-r>=0 and p-r<t_max)],[lab for lab,p in zip(label,point) if p-r>=0 and p-r<t_max])

def pad(feat,point,label, min_len):
    if np.shape(feat)[0] <= min_len:
       return (np.pad(feat, ((0,min_len-np.shape(feat)[0]), (0,0)), mode='constant', constant_values=0),point,label)
    else:
       return (feat,point,label)

def process_feat(feat,point,label, length):

    if len(feat) > length:
        # print(point)
        return random_extract(feat,point,label,length)
    else:
        return pad(feat,point,label, length)

def write_to_file(dname, dmap, cmap, itr):
    fid = open(dname + '-resultsAdd.log', 'a+')
    string_to_write = str(itr)
    for item in dmap:
        string_to_write += ' ' + '%.2f' %item
    string_to_write += ' ' + '%.2f' %cmap
    #string_to_write += ' ' + '%.2f' %cmap1
    fid.write(string_to_write + '\n')
    fid.close()

def write_seg(dname,cas,action,point,pre,seg,sim,thr,fine,segScore):
    fid = open(dname + '-result.log','a+')
    string_to_write = ''
    for c in cas:
      string_to_write += str(c)+' '
    string_to_write += '+'
    for a in action:
      string_to_write += str(a) + ' '
    string_to_write += '+'
    string_to_write += str(point) + '+'
    # print(pre)
    for p in pre:
      string_to_write += str(p) + ' '
    string_to_write += '+'
    for s in seg:
      string_to_write += str(s) + ' '
    string_to_write += '+'
    for s in sim:
      string_to_write += str(s) + ' '
    string_to_write += '+' + str(thr)
    
    string_to_write += '+'
    for s in fine:
      string_to_write += str(s) + ' '

    string_to_write += '+'
    for s in segScore:
      string_to_write += str(s) + ' '
    fid.write(string_to_write + '\n')
    fid.close()
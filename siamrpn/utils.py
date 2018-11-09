import torch

def compute_mat_dist(a,b,squared=False):
    """
    Input 2 embedding matrices and output distance matrix
    """
    assert a.size(1) == b.size(1)
    dot_product = a.mm(b.t())
    a_square = torch.pow(a, 2).sum(dim=1, keepdim=True)
    b_square = torch.pow(b, 2).sum(dim=1, keepdim=True)
    dist = a_square - 2*dot_product + b_square.t()
    dist = dist.clamp(min=0)
    if not squared:
        epsilon = 1e-12
        mask = (dist.eq(0))
        dist += epsilon * mask.float()
        dist = dist.sqrt()
        dist *= (1-mask.float())
    return dist

def bb_intersection_over_union(boxA, boxB):
    """
    copied from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

def computeIOU_torch(boxA, boxB):
    """
    Input:
        - boxA: (m, 4) torch.Tensor
        - boxB: (n, 4) torch.Tensor
    Intermediate Variables:
        - xA, yA, xB, yB, interArea: (m, n)
        - boxAArea: (m)
        - boxBArea: (n)
    Output:
        - iou: (m, n)
    """
    xA = torch.max(boxA[:,0].unsqueeze(1), boxB[:,0].unsqueeze(1).t())
    yA = torch.max(boxA[:,1].unsqueeze(1), boxB[:,1].unsqueeze(1).t())
    xB = torch.min(boxA[:,2].unsqueeze(1), boxB[:,2].unsqueeze(1).t())
    yB = torch.min(boxA[:,3].unsqueeze(1), boxB[:,3].unsqueeze(1).t())

    interArea = (xB-xA+1).clamp(min=0) * (yB-yA+1).clamp(min=0)
    
    boxAArea = (boxA[:,2]-boxA[:,0]+1) * (boxA[:,3]-boxA[:,1]+1)
    boxBArea = (boxB[:,2]-boxB[:,0]+1) * (boxB[:,3]-boxB[:,1]+1)

    iou = interArea / (boxAArea.unsqueeze(1) + boxBArea.unsqueeze(1).t() - interArea).float()
    return iou

def x1y1x2y2_to_xywh(gtbox):
    return list(map(round, [(gtbox[0]+gtbox[2])/2., (gtbox[1]+gtbox[3])/2., gtbox[2]-gtbox[0], gtbox[3]-gtbox[1]]))

def xywh_to_x1y1x2y2(gtbox):
    return list(map(round, [gtbox[0]-gtbox[2]/2., gtbox[1]-gtbox[3]/2., gtbox[0]+gtbox[2]/2., gtbox[1]+gtbox[3]/2.]))

def x1y1wh_to_xywh(gtbox):
    x1, y1, w, h = gtbox
    return [round(x1 + w/2.), round(y1 + h/2.), w, h]

def x1y1wh_to_x1y1x2y2(gtbox):
    x1, y1, w, h = gtbox
    return [x1, y1, x1+w, y1+h]

def clip_anchor(x1y1x2y2, size):
    return list(map(lambda num: min(size, max(0, num)), x1y1x2y2))

def xywh_to_x1y1x2y2_torch(xywh):
    """
    Input: 
        xywh: (N, k, 4, 17, 17)
        transform in-place!
    """
    xywh[:,:,0] = xywh[:,:,0] - xywh[:,:,2]/2
    xywh[:,:,1] = xywh[:,:,1] - xywh[:,:,3]/2
    xywh[:,:,2] = xywh[:,:,0] + xywh[:,:,2]
    xywh[:,:,3] = xywh[:,:,1] + xywh[:,:,3]

def get_anchors(k, grid_len, detection_size, anchor_shape, num_grids, cuda=False):
    """
    Output:
        anchors: torch Tensor (1, k, 4, 17, 17)
    """
    anchors = torch.zeros((1, k, 4, num_grids, num_grids))
    if cuda:
        anchors = anchors.cuda()
    for a in range(num_grids):
        for b in range(num_grids):
            for c in range(k):
                anchor = [grid_len//2+grid_len*a, grid_len//2+grid_len*b, anchor_shape[c][0], anchor_shape[c][1]]
                anchor_x1y1x2y2 = xywh_to_x1y1x2y2(anchor)
                anchor_x1y1x2y2 = clip_anchor(anchor_x1y1x2y2,detection_size)
                anchor = x1y1x2y2_to_xywh(anchor_x1y1x2y2)
                anchors[0,c,:,a,b] = torch.Tensor(anchor).cuda()
    return anchors

def regression_adjust(routput, anchors):
    """
    Input:
        anchors: (1, k, 4, 17, 17)  xywh
        routput: (N, k, 4, 17, 17)  xywh
    Output:
        bboxes: (N, k, 4, 17, 17) 
    """
    bboxes = torch.zeros(routput.shape).cuda()
    bboxes[:,:,0] = routput[:,:,0]*anchors[:,:,2] + anchors[:,:,0]
    bboxes[:,:,1] = routput[:,:,1]*anchors[:,:,3] + anchors[:,:,1]
    bboxes[:,:,2] = torch.exp(routput[:,:,2])*anchors[:,:,2]
    bboxes[:,:,3] = torch.exp(routput[:,:,3])*anchors[:,:,3]
    xywh_to_x1y1x2y2_torch(bboxes)
    # bboxes = bboxes.long()
    return bboxes

def nms(boxes, scores, overlap=0.5, top_k=200):
    """
    Input:
        - boxes: (-1, 4)  x1y1x2y2
        - scores: (-1, )
        - overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        - top_k: (int) The Maximum number of box preds to consider.
    """
    # keep = scores.new(scores.size(0)).zero_().long()
    keep = torch.zeros(scores.shape).long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()  # tensor.new() return a new tensor of same data type within same device
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count



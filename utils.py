def get_detections(net, blob):
    net.setInput(blob)
    boxs, masks = net.forward(['detection_out_final', 'detection_masks'])
    return boxs, masks
def detect(model, frame):
    
    result = model.predict(frame, conf = 0.4, iou = 0.5)
    boxes = result[0].boxes
    bounding_box_list = []
    for i in range(len(boxes)):
        box = [x1, y1, x2, y2] = boxes[i].xyxy.cpu().numpy()[0]
        bounding_box_list.append(box)
    return boxes, bounding_box_list
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import non_max_suppression


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load('yolov5s.pt', map_location=device)  # model weight here, replace yolov5s.pt
stride = int(model.stride.max()) 
cudnn.benchmark = True

names = model.module.names if hasattr(model, 'module') else model.names

cap = cv2.VideoCapture(0) # source: replace the 0 for other source.
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        img = torch.from_numpy(frame)
        img = img.permute(2, 0, 1 ).float().to(device)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.28, 0.45) # img, conf, iou

        for i, det in enumerate(pred):
            if len(det): 
            	for d in det: # d = (x1, y1, x2, y2, conf, cls)
	                x1 = int(d[0].item())
	                y1 = int(d[1].item())
	                x2 = int(d[2].item())
	                y2 = int(d[3].item())
	                conf = round(d[4].item(), 2)
	                c = int(d[5].item())
	                
	                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2) # box
	                frame = cv2.putText(frame, names[c], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA) # object name
	                print(f'{x1} {x2} {y1} {y2} {conf} {c}') # print (x1, y1, x2, y2, conf, cls)


        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
  
    else:
        break


cap.release()
cv2.destroyAllWindows()

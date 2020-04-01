# flake8: noqa
#%%
import torch

from forgery_detection.data.set import FileList
from forgery_detection.data.utils import resized_crop

f = FileList.load(
    "/mnt/ssd1/sebastian/file_lists/c40/"
    "youtube_Deepfakes_Face2Face_FaceSwap_NeuralTextures_c40_face_images_tracked_100_100_8.json"
)
unnormalize = lambda x: x * torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(
    1
) + torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1)
d = f.get_dataset(
    "train",
    image_transforms=resized_crop(229),
    tensor_transforms=[unnormalize],
    should_align_faces=True,
)

class_idx_to_label = """0: unlabeled
1: person
2: bicycle
3: car
4: motorcycle
5: airplane
6: bus
7: train
8: truck
9: boat
10: traffic light
11: fire hydrant
12: street sign
13: stop sign
14: parking meter
15: bench
16: bird
17: cat
18: dog
19: horse
20: sheep
21: cow
22: elephant
23: bear
24: zebra
25: giraffe
26: hat
27: backpack
28: umbrella
29: shoe
30: eye glasses
31: handbag
32: tie
33: suitcase
34: frisbee
35: skis
36: snowboard
37: sports ball
38: kite
39: baseball bat
40: baseball glove
41: skateboard
42: surfboard
43: tennis racket
44: bottle
45: plate
46: wine glass
47: cup
48: fork
49: knife
50: spoon
51: bowl
52: banana
53: apple
54: sandwich
55: orange
56: broccoli
57: carrot
58: hot dog
59: pizza
60: donut
61: cake
62: chair
63: couch
64: potted plant
65: bed
66: mirror
67: dining table
68: window
69: desk
70: toilet
71: door
72: tv
73: laptop
74: mouse
75: remote
76: keyboard
77: cell phone
78: microwave
79: oven
80: toaster
81: sink
82: refrigerator
83: blender
84: book
85: clock
86: vase
87: scissors
88: teddy bear
89: hair drier
90: toothbrush
91: hair brush
92: banner
93: blanket
94: branch
95: bridge
96: building-other
97: bush
98: cabinet
99: cage
100: cardboard
101: carpet
102: ceiling-other
103: ceiling-tile
104: cloth
105: clothes
106: clouds
107: counter
108: cupboard
109: curtain
110: desk-stuff
111: dirt
112: door-stuff
113: fence
114: floor-marble
115: floor-other
116: floor-stone
117: floor-tile
118: floor-wood
119: flower
120: fog
121: food-other
122: fruit
123: furniture-other
124: grass
125: gravel
126: ground-other
127: hill
128: house
129: leaves
130: light
131: mat
132: metal
133: mirror-stuff
134: moss
135: mountain
136: mud
137: napkin
138: net
139: paper
140: pavement
141: pillow
142: plant-other
143: plastic
144: platform
145: playingfield
146: railing
147: railroad
148: river
149: road
150: rock
151: roof
152: rug
153: salad
154: sand
155: sea
156: shelf
157: sky-other
158: skyscraper
159: snow
160: solid-other
161: stairs
162: stone
163: straw
164: structural-other
165: table
166: tent
167: textile-other
168: towel
169: tree
170: vegetable
171: wall-brick
172: wall-concrete
173: wall-other
174: wall-panel
175: wall-stone
176: wall-tile
177: wall-wood
178: water-other
179: waterdrops
180: window-blind
181: window-other
182: wood""".split(
    "\n"
)[
    0:93
]

#%%
import matplotlib.pyplot as plt

# for i in range(10):
#     print(d[22551 + i * 42, 22551 + i * 42][0].shape)
#     print(d[1178494 + i * 42, 1178494 + i * 42][0].shape)

img_pristine = torch.cat(
    [d[22551 + i * 42, 22551 + i * 42][0] for i in range(10)], dim=2
)
img_nt = torch.cat([d[1178494 + i * 42, 1178494 + i * 42][0] for i in range(10)], dim=2)
res_img = torch.cat((img_pristine, img_nt), dim=1).permute((1, 2, 0))
plt.imshow(res_img)
plt.show()
# fig = plt.figure()
# for i in range(10):
#     fig.add_subplot(1, 10, i + 1)
#     plt.imshow(d[22551 + i * 42, 22551 + i * 42][0].permute((1, 2, 0)))
# plt.show()
# fig = plt.figure()
# for i in range(10):
#     fig.add_subplot(1, 10, i+1)
#     plt.imshow(d[1178494 + i * 42, 1178494 + i * 42][0].permute((1, 2, 0)))
# plt.show()


#%%
from torchvision.models.detection import maskrcnn_resnet50_fpn

m = maskrcnn_resnet50_fpn(pretrained=True).eval().cuda(2)

#%%
min_length = 423
prestine_video_start = 22551
neural_textures_start = 1178494

from torchvision.transforms import ToTensor
from tqdm import tqdm
import torch


def get_lable_scores(dataset, start, end):
    label_scores = []
    i = 0
    for idx in tqdm(range(start, end)):
        img = dataset[idx, idx][0].unsqueeze(0).cuda(2)
        out = m(img)[0]
        labels = out["labels"].detach().cpu()
        scores = out["scores"].detach().cpu()
        label_scores += [(labels, scores)]

    return label_scores


pristine_label_scores = get_lable_scores(
    d, prestine_video_start, prestine_video_start + min_length
)
fakerones_A_label_scores = get_lable_scores(
    d, neural_textures_start, neural_textures_start + min_length
)

#%%
pristine_label_length = [len(x[0]) for x in pristine_label_scores]
fakerones_A_label_length = [len(x[0]) for x in fakerones_A_label_scores]

plt.hist(pristine_label_length)
plt.hist(fakerones_A_label_length)
plt.show()

#%%
pristine_all_label = torch.cat([x[0] for x in pristine_label_scores])
fakerones_A_all_labels = torch.cat([x[0] for x in fakerones_A_label_scores])

plt.hist(pristine_all_label)
# plt.show()
plt.hist(fakerones_A_all_labels)
plt.show()

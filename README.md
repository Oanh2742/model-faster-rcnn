# model-faster-rcnn
Enter your gg drive
Creat a new folder name "AI_COLAB" on your gg drive
Upload folder object_detection to "AI_COLAB" folder
Open Google Colaboratory from drive, then a new .ipynb will be created in a new tab (you can refer to https://vi-vn.facebook.com/notes/ai-viet-nam/t%E1%BB%95ng-quan-v%E1%BB%81-google-colab/354433738517097/)
To open gg drive in colab, type 2 lines:
	from google.colab import drive
	drive.mount('/content/drive/')
in code cell, then run it. Do like description to complete this step. Then press Ctrl + M + M to transform this code to text.
Open new code cell (Ctrl + M + B) then type:
	%cd /content/drive/My\ Drive/AI_COLAB/object_detection/
	!python tracking_object.py
to run this file.

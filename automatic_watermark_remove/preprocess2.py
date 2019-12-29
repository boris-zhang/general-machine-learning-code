import sys, os
import cv2
import numpy as np


def preprocess(srcfolder, dest_folder, sizew=700, sizeh=467):

	srcfd = os.path.abspath(srcfolder)
	destfd = os.path.abspath(dest_folder)

	if not (os.path.exists(srcfd) or os.path.exists(destfd)) :
		print ("Directory %s or %s doesn't exist." % (srcfd, destfd))
		return None
	i = 0
	for root, dirs, files in os.walk(srcfolder):
		for file in files:
			path = (os.sep.join([os.path.abspath(root), file]))
			img = cv2.imread(path)
			if img is not None:
				m, n, p = img.shape
				print('m, n, p', m, n, p, sizew, sizeh)
				m_t, n_t = int((sizeh-m)/2), int((sizew-n)/2)
				print('m_t, n_t: ', m_t, n_t)
				final_img = np.pad(img, ((m_t, sizeh-m-m_t), (n_t, sizew-n-n_t), (0, 0)), mode='constant')
				cv2.imwrite(os.sep.join([dest_folder, file]), final_img)
				print("Saved to : %s %s" % (file, final_img.shape))
				i +=1
	print("    %d files have been processed." % i)

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Format : %s <srcfolder>  %s <dest_folder>"%(sys.argv[0], sys.argv[1]))
	else:
		preprocess(sys.argv[0], sys.argv[1])
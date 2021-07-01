#Importing the required libraries
import cv2 as cv
import os


#Lists to store the names of the Invoices and the Logos
invoices = []
logos = []

#Filling the invoices and logos lists with the corresponding names 
for tempImg in os.listdir("Invoice Image"):
    invoices.append(tempImg)    
    
for tempImg in os.listdir("Logo"):
    logos.append(tempImg) 

#Dictionary to store logos and corresponding invoices
logoDictionary = {}

#Each logo is mapped to an empty list to handle multiple invoices with same logo
for tempLogo in logos:
    logoDictionary[tempLogo[0:-4]] = []


#This nested for loop categorizes the innvoices based on logos
#Each invoice image is matched with each logo present in the Logo directory. 
for tempInvoice in invoices:
    for tempLogo in logos:
        
        #Reading the invoice invoice from the 'Invoice Image' directory 
        invoicePath = os.path.join("Invoice Image", tempInvoice)
        invoice = cv.imread(invoicePath)
        
        #Reading the invoice invoice from the 'Logo' directory
        logoPath = os.path.join("Logo", tempLogo)
        logo = cv.imread(logoPath)
        
        #Storing the width and height of the template logo
        w = logo.shape[1]
        h = logo.shape[0]
        
        #The cv.TM_SQDIFF_NORMED method is used to match the logo in the invoice
        method = eval('cv.TM_SQDIFF_NORMED')
        
        #The cv.matchTemplate() method mathes the logo with the invoice ans the result is stored in res
        res = cv.matchTemplate(invoice, logo, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        
        #The min_loc list stores the coordinate of the top left corner of the matched region in the invoice image 
        top_left = min_loc
        #The bottom_right list stores the coordinates of the bottom right corner of the matched region in the invoice image 
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        #The top_left and bottom_right coordinates are used to crop the matched region from the invoice image
        croppedInvoice = invoice[top_left[1]:bottom_right[1], 
                             top_left[0]:bottom_right[0]]
        
        
        #The SIFT algorithm extracts the keypoints and computes its descriptors.
        #This algorithm is used to extract the keypoints and the descriptors of the  cropped image and the template logo
        #This will be used to compare the features of both the images to check how similar they are.
        sift = cv.SIFT_create()
    
        #Extracting the key points and descriptors of the cropped image and the template logo.
        kp_1, desc_1 = sift.detectAndCompute(logo, None)
        kp_2, desc_2 = sift.detectAndCompute(croppedInvoice, None)
        
        #FLANN algorithm has been used for feature matching.
        #Two dictionaries have to be passed which specifies the algorithm to be used and its related parameters.
        
        #The first dictionary is index_params which contains the algorithm to be used and the number of threes.
        #For SIFT algorithm the parameters are algorithm = 1, trees = 5.
        index_params = dict(algorithm = 1, trees = 5)
        #The second dictionary is search_params which specifies the number of times the trees in the index_params should be recursively traversed.  
        search_params = dict(checks = 100)  
        
        #Matching the cropped invoice image and the template logo 
        flann = cv.FlannBasedMatcher(index_params, search_params)
        #The matches found beween both the images are stored in the 'matches' list. 
        matches = flann.knnMatch(desc_1, desc_2, 2)
        
        #Not all the matches found are good matches.
        #Finding the number of good points from all the matches
        good_points = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good_points.append(m)
        
        #Finding the similarity between the cropped invoice image and the template logo. 
        #If the similarity is greater than 85% then the invoice added into the logo dictionary to the corresponding template logo.
        if (len(good_points)/len(matches)) * 100 >= 85:
            logoDictionary[tempLogo[0:-4]].append(tempInvoice);

print()

#Printing the final logo dictionary
for key in logoDictionary:
    print(key.capitalize() + " - " , end = ' ')
    none = 1
    for value in logoDictionary[key]:
        none = 0
        print(value[:-4], sep = "," ,end  = ' ')
    if none == 1:
        print("No invoices matched to this logo", end = " ")
    print()

import  cv2
import matplotlib.pylab as mp
import numpy as np
import datetime as dt
import pyttsx3
engine = pyttsx3.init()



def region_of_inter(img,vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image  = cv2.bitwise_and(img,mask)
    return masked_image


def draw_line_image(img, lines):
    #making the copy of imaeg
    #the reason it was not showing the lane was that we left parmeter of np.copy emplty so it was not showing the lanes
    img = np.copy(img)
    #we haave  left the papmeter blank in img.shpae
    blank_imgae = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    #inderting over each item
    for line in lines:
        for x1,y1,x2,y2 in line:
            #here it was drwaing the line which is nor vertical but it was drawn along the y axis as we in pt2 we passed x1 insted of x2
            cv2.line(blank_imgae,(x1,y1), (x2,y2), (0,0,225),thickness=10)
    #the one error was herhe taht we give more weight to blacnkimge so we were not able to see the image
    img = cv2.addWeighted(img, 0.8, blank_imgae, 1, 0.0)
    return img

def lane_draw(imgae):
    #at some time after in the video that it was showing error that it was not getting the height
    try:
        height = imgae.shpae[0]
        width = imgae.shape[1]
    except AttributeError:
        pass

    region_of_intrest_ver =[
            (0, 384),
            (183, 245),
            (470,245),
            (640, 340)
    ]
    try :
        blur_image =cv2.GaussianBlur(imgae, (5,5))
        #canny imaaegs
        canny_img = cv2.Canny(blur_image,100,200)
    except:
        canny_img = cv2.Canny(imgae, 100, 200)

    #image with masked of polygone
    masked_image = region_of_inter(canny_img, np.array([(region_of_intrest_ver)], np.int32), )

    #praoblistic hough line trandform
    p_line = cv2.HoughLinesP(masked_image, rho=6, theta=np.pi/180, threshold=140, minLineLength=30, maxLineGap=10)

    #images with lines
    img_with_lanes = draw_line_image(imgae, p_line)
    return img_with_lanes


def speed(image):
    for x,y,w,h in car_detect:
        #one error was herhe that we ahev given the white spcae between sum_Y
        sum_y = np.sum(y)
        speed = float(y/60)
        speed1 = round(speed)
        speed2 = str((speed1))
        text =f'{speed2}pxpersec'
        cv2.putText(image,text,(x-w,y+w), cv2.FONT_ITALIC,0.5,(225,225,225),2)
    return image


#saying cv2 to to reconize mt file of video
video = cv2.VideoCapture('car5.mp4')

#callsifyng pre trained algorithm
car_detect_file = 'cars_op.xml'
body_detect_file = 'haarcascade_fullbody.xml'

while video.isOpened():
    #reaing each fraem in video
    op_sucess,frame = video.read()
    try :
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    except:
        pass

    # crareing bank image for a peroson o screee link
    blank_imgae = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    #cascade clasfify
    car_classify = cv2.CascadeClassifier(car_detect_file)
    body_pedi_callify = cv2.CascadeClassifier(body_detect_file)

    #sayign it to dectcte multisale
    car_detect = car_classify.detectMultiScale(frame)
    #the errror was here that we use the body_detect_file insted of using the body_pedi_calsify
    body_detect =body_pedi_callify.detectMultiScale(frame)

    #putting live date and time o frame
    #the one error was here brcuse here we have not converted it to str and we were using it as the int
    date_time = str(dt.datetime.now())
    cv2.putText(frame, date_time, (10,10), cv2.FONT_ITALIC, 0.5, (225, 0, 0), 2)
    cv2.putText(blank_imgae, date_time, (10, 40), cv2.FONT_ITALIC, 0.5, (225, 0, 0), 2)


    #priting the shape of image
    height = frame.shape[0]
    width = frame.shape[1]

    no_of_car = 0

    for x,y,w,h in car_detect:

        #COUNTING NO. OF CAR MADE BY PARAM
        no_of_car = no_of_car + 1
        no_of_car_s = f'no.of car around you = {no_of_car}'
        text = f'{no_of_car},"VEHICLE"'


        # showing the no. of car and and puuting text on top left coner of recatngle
        cv2.putText(frame, text, (x, y), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)

        # putting txt on recatangle
        cv2.putText(frame, no_of_car_s, (15, 30), cv2.FONT_ITALIC, 0.8, (225, 0, 0), 2)
        cv2.putText(blank_imgae, no_of_car_s, (15, 60), cv2.FONT_ITALIC, 0.8, (225, 0, 0), 2)

        #drawing recatngle around it
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 3)

        #dectCing that if car is overtaking
        X_co =x+w
        Y_co =y+h
        if X_co >470 or 183 and Y_co>245 :
            cv2.putText(frame, "car overtaking  ", (250, 300), cv2.FONT_ITALIC, 1, (0,0,0), 2)#we get the co ordiante form matplot.pylab
            cv2.putText(blank_imgae, "car is overtaking miantain your speed ", ((30), round(height/2)), cv2.FONT_ITALIC, 1, (0, 0, 225), 2)
        else:
            pass

        if no_of_car> 4:
            # puttign text if anyone does not here sound
            cv2.putText(blank_imgae, "Be Careful there are more car around you ", ((20), round(height / 3)),
                        cv2.FONT_ITALIC, 1, (0, 0, 225), 2)
        else:
            pass


    #fugruting out the avg picel coverd in 1 sec speed of a car
    spped_of_car = speed(frame)

    for x,y,w,h in body_detect:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 225, 224), 3)
        cv2.putText(frame, "OBSTALE", (x, y), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)

    #PRITNNG CO-ORDINATE OF ACR AND BOVY OR PEDISTARAIN
    print(car_detect)
    print(body_detect)

    #the one error was here becuse at some point it does not detct line so it was shwing error
    try:
        frame = lane_draw(frame)
    except TypeError:
        cv2.putText(frame,"no lane dectedc", (100, 700), cv2.FONT_HERSHEY_DUPLEX,3,(0,225,0),thickness=2)

     #VOIC ASSTING USING THE PYTTSX3 MADE BY SHAUARY
    if no_of_car > 5:
        #puttign text if anyone does not here sound
        engine.say("Be Careful there are more than five car aroudn u ")
        engine.runAndWait()
    else:
         pass
    #showimg the video in winodw uaing cv2 method called imshow
    cv2.imshow('tesla_backend', frame)
    cv2.imshow('screen', blank_imgae)

    #matplotlib window
    # mp.imshow(frame)
    # mp.show()

    #adding dealy to image
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
print("code complted")

#timeline of our code
#10:15 till 11:15 we write teh line from 1 to 51
#from 1 to 51 two people of our team has written the code me and shaury
#other all the lines of code take around the 1 hours to 30 min from (1:00 pm to 2:00pm and from 3:00pm to 3:30pm )
# it took us 2. 30  hours in total  to write the code but it took 1 debug the code()
#totla time to write teh code ios 3 hours and code is written b 2 perosn of our team
#it took around 45 min to debuig ourr code whihc is from which is from (4:45 to 5:37)
#it ws not showing the lane when we run the project and it took us to 40 min to sort that error (as 5:40 we were haveing the stem ai workshop so we started to work on lane at 7:10 and to 7:40
# it took 40 min sort the lane error  )
#in total we can say that we took 4 hours to completehe code including debbuging
#WE ADED THE VOICE ASSTING AND COUNTUG THE NUMBER OF CAR IN FRAME IT TOOK US AROUND AN HOUR E STARETD AT (10:00 PM AND FINISHED AT 12:10 AM )

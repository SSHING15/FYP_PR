# ====== IMPORTING MODULES
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
import sys
sys.path.append('../../')
import pydic

import cv2
import numpy as np

#_____________________ANNOTATE HIGH LOAD AREAS_____________________________________
def resize_image(image, width):
    """Resize image to a specified width while maintaining aspect ratio."""
    aspect_ratio = width / float(image.shape[1])
    height = int(image.shape[0] * aspect_ratio)
    return cv2.resize(image, (width, height))

def get_16_9_bounding_box(x, y, w, h):
    """Adjust bounding box to have a 16:9 aspect ratio while ensuring it covers the original area.
    The input width and height are treated as minimum values."""
    center_x, center_y = x + w // 2, y + h // 2
    aspect_ratio = 16 / 9 # can change if needed 

    # Determine new dimensions based on aspect ratio and ensure they are at least as large as the original dimensions
    if w / h > aspect_ratio:
        new_h = max(h, w / aspect_ratio)
        new_w = new_h * aspect_ratio
    else:
        new_w = max(w, h * aspect_ratio)
        new_h = new_w / aspect_ratio

    # Adjust the top-left corner to ensure the new dimensions are centered around the blob
    x = center_x - new_w // 2
    y = center_y - new_h // 2

    return int(x), int(y), int(new_w), int(new_h)

def annotate_red_blobs(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for the color red in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([8, 255, 255])
   

    # Create masks for red color
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask = mask1 #| mask2

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare to draw rectangles, X markers, and print coordinates
    output_image = image.copy()

    # Change these variable
    #camera_distance=30 #30cm defult distance from camera to material
    centre_location=[162.5,162.5,340]
    minimum_loc=[25,25,45] #need to change z (inline with hieght so material)
    maximum_loc=[300,300,340]

    coordinates = []

    for contour in contours:
        # Compute the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Ensure the rectangle is a 16:9 aspect ratio and at least as large as the original dimensions
        x, y, w, h = get_16_9_bounding_box(x, y, w, h)

        # Draw the rectangle on the image
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Calculate center of the rectangle
        center_x = x + w // 2
        center_y = y + h // 2

        # Draw a red X at the center of the rectangle
        cross_size = 10
        cv2.line(output_image, (center_x - cross_size, center_y - cross_size), (center_x + cross_size, center_y + cross_size), (0, 0, 255), 2)
        cv2.line(output_image, (center_x - cross_size, center_y + cross_size), (center_x + cross_size, center_y - cross_size), (0, 0, 255), 2)

        # Print center coordinates and dimensions
        print(f"Center of Box: ({center_x}, {center_y}), Width: {w}, Height: {h}")

        # x_change=center_x-img_width/2 
        # y_change=center_y-img_height/2
        #new_z=(centre_location[2]-minimum_loc[2])*w/img_width+minimum_loc[2]
        new_z=np.interp(w, [0, img_width], [minimum_loc[2], maximum_loc[2]])
        new_x=np.interp(center_x, [0, img_width], [minimum_loc[0], maximum_loc[0]])
        new_y=np.interp(center_y, [0,img_height], [minimum_loc[1], maximum_loc[1]])

        #print(f"New coordinate (x,y,z): ({round(new_x,3)},{round(new_y,3)},{round(new_z,3)})")
        new_coordinates = (round(new_x, 3), round(new_y, 3), round(new_z, 3))
        coordinates.append(new_coordinates)

        # Print the new coordinates
        print(f"New coordinate (x, y, z): {new_coordinates}")
        
    # Save the annotated image
    cv2.imwrite(output_path, output_image)

    # Resize images for display
    display_width = 960  # Set a fixed width for display
    resized_original = resize_image(image, display_width)
    resized_mask = resize_image(mask, display_width)
    resized_annotated = resize_image(output_image, display_width)

    # Create windows with adjustable size
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Annotated Image', cv2.WINDOW_NORMAL)

    # Display the images
    cv2.imshow('Original Image', resized_original)
    #cv2.imshow('Mask', resized_mask)
    cv2.imshow('Annotated Image', resized_annotated)

    # Wait for a key press and close the image windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return coordinates
# Is given pixel distance
#Returns movment in cm


#_________________________________________PRINTER MOTION CONTROL___________________________________-
import serial
import time

# Initialize the serial connection to the 3D printer
printer_serial = serial.Serial('COM5', 115200, timeout=2)
time.sleep(2)  # Allow time for the connection to initialize

def send_gcode_command(command):
    """Send a G-code command to the 3D printer and wait for response"""
    printer_serial.write((command + '\n').encode())  # Send command with newline
    response = printer_serial.readline().decode().strip()
    print("Printer response:", response)
    
    while response.lower() != 'ok':
        response = printer_serial.readline().decode().strip()
        print("Printer response:", response)

def move_to_coordinates(x, y, z, feed_rate=3000):
    """Move the 3D printer to the specified coordinates (X, Y, Z)"""
    gcode_command = f"G0 X{x} Y{y} Z{z}" #F{feed_rate}"  # G-code for movement
    send_gcode_command(gcode_command)

def home_printer():
    """Home the 3D printer (move to the origin)"""
    send_gcode_command("G28")  # G-code to home all axes    


#________________________________STEPPER MOTOR CONTROL________________
def stepper_initalise():
    with open('Gcode_file.txt', 'r') as file:
        # Read all lines and store them as a list of strings
        lines = file.readlines()

    # Each line in the file is now a string element in the 'lines' list
    for line in lines:
        print(line.strip())  # Use .strip() to remove any trailing newlines
        send_gcode_command(line.strip())

def stepper_step():
    send_gcode_command("G0 G91 E-5")

#__________________________________DIC ANALYSIS_________________________________________
def DIC_analysis(DIC_number):
    #  ====== RUN PYDIC TO COMPUTE DISPLACEMENT AND STRAIN FIELDS (STRUCTURED GRID)
    correl_wind_size = (80,80) # the size in pixel of the correlation windows
    correl_grid_size = (20,20) # the size in pixel of the interval (dx,dy) of the correlation grid

    # Read image series and write a separated result file 
    # pydic.init('./img_3/*.tif', correl_wind_size, correl_grid_size, "result.dic")
    pydic.init(f'./DIC_folder/DIC_{DIC_num}/DIC_images_{DIC_number}/*.tif', correl_wind_size, correl_grid_size, "result.dic")

    # Read the result file for computing strain and displacement field from the result file 
    pydic.read_dic_file('result.dic', interpolation='spline', strain_type='cauchy', save_image=True, scale_disp=10, scale_grid=25, meta_info_file='img/meta-data.txt')

    # ======= STANDARD POST-TREATMENT : STRAIN FIELD MAP PLOTTING
    # The pydic.grid_list is a list that contains all the correlated grids (one per image)
    # The grid objects are the main objects of pydic  
    last_grid = pydic.grid_list[-1]  
    
    dir_path = f"DIC_folder/DIC_{DIC_num}/DIC_output_{DIC_num}"
    
    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)
    last_grid.plot_field(last_grid.strain_xx, 'xx strain')
    plt.savefig(f'{dir_path}/strain_xx_{DIC_number}.png', dpi=300)

    last_grid.plot_field(last_grid.strain_yy, 'yy strain')
    plt.savefig(f'{dir_path}/strain_yy_{DIC_number}.png', dpi=300)

    last_grid.plot_field(last_grid.strain_xy, 'xy strain')
    plt.savefig(f'{dir_path}/strain_xy_{DIC_number}.png', dpi=300)

    plt.close()


import os
from picamera2 import Picamera2
import numpy as np
import time
import cv2

import RPi.GPIO as GPIO
from hx711 import HX711  # import the class HX711
import time

#____________________ SET UP PINS________________________
# Set up GPIO and HX711
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  # set GPIO pin mode to BCM numbering
hx = HX711(dout_pin=5, pd_sck_pin=6)  # create an HX711 object

#_____________________________LOAD CELL__________________________
def load_cell_value():
    # Get weight in kg
    weight_kg = hx.get_weight_mean(5)
    print(f"Weight: {weight_kg:.2f} kg")
    return weight_kg
    # Wait a little before the next reading
    #time.sleep(0.5)

def load_cell_cal():
    # Tare the load cell (zero out the scale)
    hx.zero()
    print("Tare done! Remove any weight from the load cell.")

    # Allow some time to stabilize
    time.sleep(2)

    # Get raw readings for tare
    tare_value = hx.get_data_mean()
    print(f"Tare Value: {tare_value}")

    # Prompt user to place a known weight on the load cell
    input("Place a known weight on the load cell and press Enter...")

    # Get raw readings with known weight
    known_weight_value = hx.get_data_mean()
    print(f"Known Weight Value: {known_weight_value}")

    # Enter the known weight in kg
    known_weight_kg = float(input("Enter the known weight in kg: "))

    # Calculate calibration factor
    calibration_factor = (known_weight_value - tare_value) / known_weight_kg
    print(f"Calibration Factor: {calibration_factor}")

    # Set calibration factor to HX711
    hx.set_scale_ratio(calibration_factor)


#________________________________IMAGE CAPTURE_________________________________-
def capture_images(DIC_number):#filename='captures/image.jpg'):
   
    dir_path = f"DIC_folder/DIC_{DIC_num}/DIC_images_{DIC_number}"
    
    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)

    # Path for the load cell readings text file
    load_values = []

    #os.makedirs('captures', exist_ok=True)
    os.environ['LIBCAMERA_LOG_LEVELS'] = '*:ERROR'

    # Initialize the camera
    picam2 = Picamera2()
    
    capture_config = picam2.create_still_configuration(main={"size": (3280, 2464)})
    picam2.configure(capture_config)

    try:
        picam2.start()
        #time.sleep(1.5)
        number_of_imgs=30 # change this to number of images we need 
        for img_number in range(number_of_imgs): 
            # Capture the image as a NumPy array
            image = picam2.capture_array()
            print(f"Image size: {image.shape}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
            image = cv2.rotate(image, cv2.ROTATE_180)

            #cv2.imwrite(f"DIC_images/DIC_images_{DIC_number}/image_{img_number}.tif", image)
            cv2.imwrite(f"{dir_path}/image_{img_number}.tif", image)
            
            load_kg=load_cell_value()
            load_values.append(load_kg)
            stepper_step()
            time.sleep(0.25)

    except Exception as e:
        print(f"An error occurred while capturing the image: {e}")
    
    finally:
        picam2.stop()
        picam2.close()

         #Save the load values to a text file
        load_file_path = f"{dir_path}/load_cell_readings_(kg)_{DIC_number}.txt"
        with open(load_file_path, 'w') as load_file:
            for load_value in load_values:
                load_file.write(f"{load_value}\n")
        print(f"Load readings saved to {load_file_path}")

    #return filename

#______________________________MAIN FUNCTION__________________________________________
try: 
    #____________________ HOME PRINTER AND MOVE CAMERA TO CENTRE____________
    print("Homing printer...")
    home_printer()
    move_to_coordinates(162.5, 162.5, 340)
    time.sleep(5) # wait so camrea is in positon
    
    #Calibrate Load cell
    load_cell_cal(hx)
    stepper_initalise()
    #_______________________ TAKE PHOTOs, get load cell readings, step stepper motor _______________________
    DIC_num=0
    capture_images(DIC_num)

    #_____________________________________ DIC ANALYSIS__________________
    DIC_analysis(DIC_num)
    
    #________________________ HIGH LOAD AREA CACLUALTOR AND NEW CAMERA POSTION COORDINATES__________________
    coordinates_array=annotate_red_blobs(f'DIC_folder/DIC_{DIC_num}/DIC_output_{DIC_num}/strain_xy_{DIC_num}.png', f'DIC_folder/DIC_{DIC_num}/DIC_output_{DIC_num}/annotated_image_{DIC_num}.png')

    #______________________#3D PRINTER MOVEMENT________________
    DIC_num+=1
    for x,y,z in coordinates_array: 
        stepper_initalise()
        move_to_coordinates(x, y, z)
        time.sleep(5)
        #_______________________ TAKE PHOTO _______________________
        capture_images(DIC_num)
        #________________________DIC___________________________
        DIC_analysis(DIC_num)
        DIC_num+=1

except KeyboardInterrupt:
    print("Program interrupted.")

finally:
    # Close the serial connection when done
    printer_serial.close()
    print("Connection closed.")

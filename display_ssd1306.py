"""
    display_ssd1306.py 

    Helper class for Adafruit_SSD1306. Displays text on 128x32 OLED display.

    Author: Mahesh Venkitachalam
    Website: electronut.in 
"""

import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

class SSD1306_Display:
    """Helper class to use OLED display"""
    def __init__(self):
        # 128x32 display with hardware I2C:
        self.disp = Adafruit_SSD1306.SSD1306_128_32(rst=None)
        # Initialize library.
        self.disp.begin()
        # Clear display.
        self.disp.clear()
        self.disp.display()
        # Create blank image for drawing.
        # Make sure to create image with mode '1' for 1-bit color.
        self.image = Image.new('1', (self.disp.width, self.disp.height))
        # Get drawing object to draw on image.
        self.draw = ImageDraw.Draw(self.image)
        # Draw a black filled box to clear the image.
        self.draw.rectangle((0,0,self.disp.width,self.disp.height), outline=0, fill=0)
        # Load default font.
        self.font = ImageFont.load_default()

    def show_txt(self, x, y, str_data, clear_display):
        """display given text"""
        # clear old data if flag set
        if clear_display:
            # Draw a black filled box to clear the image.
            self.draw.rectangle((0,0,self.disp.width, self.disp.height), 
                outline=0, fill=0)
            
        self.draw.text((x,y), str_data, font=self.font, fill=255)
        # Display image.
        self.disp.image(self.image)
        self.disp.display()
    
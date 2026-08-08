import pygame
import time

pygame.mixer.init()

pygame.mixer.music.load("alarm.mp3")
print("Audio loaded successfully!")

pygame.mixer.music.play()

print("Playing audio...")
time.sleep(10)

pygame.mixer.music.stop()
pygame.mixer.quit()

print("Done.")
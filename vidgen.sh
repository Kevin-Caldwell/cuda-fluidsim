ffmpeg -framerate 240 -i temp/u%02d.png output_u.mp4 -y & 
ffmpeg -framerate 240 -i temp/v%02d.png output_v.mp4 -y &
ffmpeg -framerate 240 -i temp/w%02d.png output_w.mp4 -y &
ffmpeg -framerate 240 -i temp/pressure%02d.png output_pressure.mp4 -y
wait
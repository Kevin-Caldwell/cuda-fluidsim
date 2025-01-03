ffmpeg -framerate 30 -i temp/u_%03d.ppm output_u.mp4 -y & 
ffmpeg -framerate 30 -i temp/v_%03d.ppm output_v.mp4 -y &
ffmpeg -framerate 30 -i temp/w_%03d.ppm output_w.mp4 -y &
ffmpeg -framerate 30 -i temp/pressure_%03d.ppm output_pressure.mp4 -y
wait
echo WATCHING: "/home/jack/chill/build/debug/modules/V4D_rtcubes/assets/shaders/avatar.meta"
inotifywait -e modify \
  '/home/jack/chill/src/v4d/modules/V4D_rtcubes/assets/shaders/avatar.glsl'\
  '/home/jack/chill/src/v4d/modules/V4D_rtcubes/base.glsl'\
  '/home/jack/chill/src/v4d/game/graphics/glsl/base.glsl'\
  '/home/jack/chill/src/v4d/game/graphics/Block.hh'\
  '/home/jack/chill/src/v4d/game/graphics/glsl/../cpp_glsl.hh'\
  '/home/jack/chill/src/v4d/modules/V4D_rtcubes/cpp_glsl.hh'\
  '/home/jack/chill/src/v4d/core/v4d.h'\
  '/home/jack/chill/src/v4d/core/utilities/graphics/shaders/cpp_glsl_head.hh'

if [[ -e '/home/jack/chill/build/debug/modules/V4D_rtcubes/assets/shaders/avatar.meta' ]] ; then
  clear
  echo "
Compiling shader...

  "
  ('/home/jack/chill/build/shadercompiler' '/home/jack/chill/src/v4d/modules/V4D_rtcubes/assets/shaders/avatar.glsl' '/home/jack/chill/build/debug/modules/V4D_rtcubes/assets/shaders/avatar.meta' '/home/jack/chill/src' '/home/jack/chill/src/v4d/core' '/home/jack/chill/src/v4d/core/utilities/graphics/shaders') && echo "[1;36m
SUCCESS
[0m" || echo "[1;31m
FAILED
[0m"
  sh -c $0 
fi

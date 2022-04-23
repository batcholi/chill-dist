echo WATCHING: "/home/jack/chill/build/debug/game/assets/shaders/fsr/rcas.meta"
inotifywait -e modify \
  '/home/jack/chill/src/v4d/game/assets/shaders/fsr/rcas.glsl'\
  '/home/jack/chill/src/v4d/game/graphics/fsr/ffx_fsr1.h'\
  '/home/jack/chill/src/v4d/game/graphics/glsl/base.glsl'\
  '/home/jack/chill/src/v4d/game/graphics/fsr/ffx_a.h'\
  '/home/jack/chill/src/v4d/game/graphics/glsl/../cpp_glsl.hh'\
  '/home/jack/chill/src/v4d/core/v4d.h'\
  '/home/jack/chill/src/v4d/core/utilities/graphics/shaders/cpp_glsl_head.hh'

if [[ -e '/home/jack/chill/build/debug/game/assets/shaders/fsr/rcas.meta' ]] ; then
  clear
  echo "
Compiling shader...

  "
  ('/home/jack/chill/build/shadercompiler' '/home/jack/chill/src/v4d/game/assets/shaders/fsr/rcas.glsl' '/home/jack/chill/build/debug/game/assets/shaders/fsr/rcas.meta' '/home/jack/chill/src' '/home/jack/chill/src/v4d/core' '/home/jack/chill/src/v4d/core/utilities/graphics/shaders') && echo "[1;36m
SUCCESS
[0m" || echo "[1;31m
FAILED
[0m"
  sh -c $0 
fi

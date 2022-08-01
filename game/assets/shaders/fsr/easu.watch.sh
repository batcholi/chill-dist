echo WATCHING: "/home/olivier/projects/chill/build/debug/game/assets/shaders/fsr/easu.meta"
inotifywait -e modify \
  '/home/olivier/projects/chill/src/v4d/game/assets/shaders/fsr/easu.glsl'\
  '/home/olivier/projects/chill/src/v4d/game/graphics/fsr/ffx_a.h'\
  '/home/olivier/projects/chill/src/v4d/game/graphics/fsr/ffx_fsr1.h'\
  '/home/olivier/projects/chill/src/v4d/game/graphics/glsl/base.glsl'\
  '/home/olivier/projects/chill/src/v4d/game/graphics/glsl/../cpp_glsl.hh'\
  '/home/olivier/projects/chill/src/v4d/core/v4d.h'\
  '/home/olivier/projects/chill/src/v4d/core/utilities/graphics/shaders/cpp_glsl_head.hh'

if [[ -e '/home/olivier/projects/chill/build/debug/game/assets/shaders/fsr/easu.meta' ]] ; then
  clear
  echo "
Compiling shader...

  "
  ('/home/olivier/projects/chill/build/shadercompiler' '/home/olivier/projects/chill/src/v4d/game/assets/shaders/fsr/easu.glsl' '/home/olivier/projects/chill/build/debug/game/assets/shaders/fsr/easu.meta' '/home/olivier/projects/chill/src' '/home/olivier/projects/chill/src/v4d/core' '/home/olivier/projects/chill/src/v4d/core/utilities/graphics/shaders') && echo "[1;36m
SUCCESS
[0m" || echo "[1;31m
FAILED
[0m"
  sh -c $0 
fi

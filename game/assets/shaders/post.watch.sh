echo WATCHING: "/home/olivier/projects/chill/build/debug/game/assets/shaders/post.meta"
inotifywait -e modify \
  '/home/olivier/projects/chill/src/v4d/game/assets/shaders/post.glsl'\
  '/home/olivier/projects/chill/src/v4d/game/graphics/glsl/base.glsl'\
  '/home/olivier/projects/chill/src/v4d/game/graphics/glsl/../cpp_glsl.hh'\
  '/home/olivier/projects/chill/src/v4d/core/v4d.h'\
  '/home/olivier/projects/chill/src/v4d/core/utilities/graphics/shaders/cpp_glsl_head.hh'

if [[ -e '/home/olivier/projects/chill/build/debug/game/assets/shaders/post.meta' ]] ; then
  clear
  echo "
Compiling shader...

  "
  ('/home/olivier/projects/chill/build/shadercompiler' '/home/olivier/projects/chill/src/v4d/game/assets/shaders/post.glsl' '/home/olivier/projects/chill/build/debug/game/assets/shaders/post.meta' '/home/olivier/projects/chill/src' '/home/olivier/projects/chill/src/v4d/core' '/home/olivier/projects/chill/src/v4d/core/utilities/graphics/shaders') && echo "[1;36m
SUCCESS
[0m" || echo "[1;31m
FAILED
[0m"
  sh -c $0 
fi

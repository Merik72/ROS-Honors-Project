#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/admin2/ros-tcp-unity/src/ROS-TCP-Endpoint"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/admin2/ros-tcp-unity/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/admin2/ros-tcp-unity/install/lib/python3/dist-packages:/home/admin2/ros-tcp-unity/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/admin2/ros-tcp-unity/build" \
    "/usr/bin/python3" \
    "/home/admin2/ros-tcp-unity/src/ROS-TCP-Endpoint/setup.py" \
    egg_info --egg-base /home/admin2/ros-tcp-unity/build/ROS-TCP-Endpoint \
    build --build-base "/home/admin2/ros-tcp-unity/build/ROS-TCP-Endpoint" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/admin2/ros-tcp-unity/install" --install-scripts="/home/admin2/ros-tcp-unity/install/bin"

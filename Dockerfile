ARG ROOT_VERSION

ARG CMAKE_CXX_STANDARD

FROM python:3.11

# WORKDIR /app
ENV LANG=C.UTF-8

COPY packages packages

RUN apt-get update -qq && \ 
    ln -sf /usr/share/zoneinfo/UTC /etc/localtime && \
    apt-get install -y $(cat packages) wget git && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/cache/apt/archives/* && \
    rm -rf /var/lib/apt/lists/*

RUN pip install numpy

ARG ROOT_VERSION=latest-stable

ARG CMAKE_CXX_STANDARD=17

RUN ROOT_GIT_URL=https://github.com/root-project/root.git \
    && if [ -z "$(git ls-remote --heads $ROOT_GIT_URL $ROOT_VERSION)" ] ; then \
    export ROOT_GIT_REVISION="v$(echo $ROOT_VERSION | cut -d. -f1)-$(echo $ROOT_VERSION | cut -d. -f2)-$(echo $ROOT_VERSION | cut -d. -f3)" \
    ; else \
    export ROOT_GIT_REVISION=$ROOT_VERSION \
    ; fi \
    # Above lines will set ROOT_GIT_REVISION to ROOT_VERSION argument if it corresponds to a valid branch name (such as ROOT_VERSION=master),
    # otherwise it will assume its a semantic version string and try to convert it into the tag format (such as 6.26.00 -> v6-26-00)
    && git clone --branch $ROOT_GIT_REVISION --depth=1 $ROOT_GIT_URL /tmp/source \
    && cd /tmp/source \
    && mkdir -p /tmp/build &&  cd /tmp/build \
    && cmake /tmp/source \
    -DCMAKE_CXX_STANDARD=$CMAKE_CXX_STANDARD \
    -Dgnuinstall=ON \
    -Dsoversion=ON \
    # For ROOT version 6.26.00 it is necessary to set `-Druntime_cxxmodules=OFF` (https://github.com/root-project/root/pull/10198)
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DCMAKE_INSTALL_BINDIR=bin \
    -DCMAKE_INSTALL_CMAKEDIR=lib/x86_64-linux-gnu/cmake/ROOT \
    -DCMAKE_INSTALL_DATAROOTDIR=share \
    -DCMAKE_INSTALL_DATADIR=share/root \
    -DCMAKE_INSTALL_DOCDIR=share/doc/root \
    -DCMAKE_INSTALL_ELISPDIR=share/emacs/site-lisp \
    -DCMAKE_INSTALL_FONTDIR=share/root/fonts \
    -DCMAKE_INSTALL_ICONDIR=share/root/icons \
    -DCMAKE_INSTALL_INCLUDEDIR=include/ROOT \
    -DCMAKE_INSTALL_JSROOTDIR=share/root/js \
    -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu \
    -DCMAKE_INSTALL_MACRODIR=share/root/macros \
    -DCMAKE_INSTALL_MANDIR=share/man \
    -DCMAKE_INSTALL_OPENUI5DIR=share/root/ui5 \
    -DCMAKE_INSTALL_PYTHONDIR=lib/python3/dist-packages \
    -DCMAKE_INSTALL_SRCDIR=/dev/null \
    -DCMAKE_INSTALL_SYSCONFDIR=/etc/root \
    -DCMAKE_INSTALL_TUTDIR=share/root/tutorials \
    && make -j$(nproc) install \
    && rm -rf /tmp/build /tmp/source

RUN pip install flux-tool 

# Set the default command to run your package
CMD [ "python", "-m", "flux-tool" ]

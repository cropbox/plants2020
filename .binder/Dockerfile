FROM tomyun/cropbox:v0.2.0

COPY . ${HOME}
USER root
RUN chown -R ${USER} ${HOME}
USER ${USER}

RUN JULIA_PROJECT=${HOME} julia -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"

#include <stdio.h>
#include <time.h>
#include <memory.h>
#include <stdlib.h>
#include <SDL.h>

#undef main

#define W 1024
#define H 1024

#define N 65000
#define MIN_INFECTION_RANGE 30.0f

short quit = 0;

static SDL_Window* w;
static SDL_Renderer* r;
static SDL_Texture* txt;
static unsigned int* people;
static unsigned int* d_people;

static unsigned int* people_pixels;
static unsigned int* d_people_pixels;

static unsigned int global_framecount = 0;

__global__ void update_people(unsigned int* d_people, int n) {
    unsigned long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx > n)
        return;
    
    unsigned int sp = d_people[idx];
    for(int i = 0; i < n; i++) {
        unsigned int p = d_people[i];
        int x = (p>>8)&2047; 
        int y = p>>19;    
        int sx = (sp>>8)&2047; 
        int sy = sp>>19;
        if(x != sx && y != sy) {
            float dist2 = (x-sx)*(x-sx) + (y-sy)*(y-sy);
            if(dist2 <= MIN_INFECTION_RANGE && p&1 && p&(1<<1)) {
                d_people[idx] |= 1;
            } 
        }
    }
}

__global__ void update_people_age(unsigned int* d_people, int n, int d_age) {
    unsigned long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx > n)
        return;

    if(d_people[idx]&(1<<1)) {
        d_people[idx] += ((d_people[idx]&1 ? 10 : 1)*d_age)<<2;
    }
    if((d_people[idx]&252)>>2 >= 63) {
        d_people[idx] &= 1073741820;
    }

}

__global__ void update_people_pixels(unsigned int* d_people, unsigned int* d_people_pixels, int n) {
    unsigned long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx > n)
        return;

    
    d_people_pixels[idx] = (((d_people[idx]&1) ? 255/(((d_people[idx]&252)>>7)+1) : 0)*(d_people[idx]&(1<<1))>>1)<<24 |
                           (((d_people[idx]&1) ? 0 : 255/(((d_people[idx]&252)>>7)+1))*(d_people[idx]&(1<<1))>>1)<<16 |
                           (255-((d_people[idx]&252)>>2)*(d_people[idx]&(1<<1))>>1)<<8 |
                           255;
}

errno_t init_sdl() {
    if(SDL_Init(SDL_INIT_VIDEO) < 0)
        return -1;
    
    w = SDL_CreateWindow("Numpu", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, W, H, SDL_WINDOW_SHOWN);
    if(w == NULL)
        return -2;
    r = SDL_CreateRenderer(w, -1, SDL_RENDERER_ACCELERATED);
    if(r == NULL)
        return -3;

    txt = SDL_CreateTexture(r, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, W, H);
    if(txt == NULL)
        return -4;
    
    return 0;

}

void render() {
    
    update_people<<< 1024, 1024 >>>(d_people, N);
    update_people_age<<< 1024, 1024 >>>(d_people, N, (global_framecount%5) ? 0 : 1);
    //update_people_pixels<<< 1024, 1024 >>>(d_people, d_people_pixels, N);

    cudaMemcpy(people, d_people, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost);
    //cudaMemcpy(people_pixels, d_people_pixels, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++) {
        if((float)(rand()/(float)RAND_MAX) < 0.01f && !(people[i]&1))
            people[i] &= ~2;
        
        unsigned int p = people[i];
        // ......11 -> alive and infected
        // ......01/0 -> dead
        // ......10 -> alive and healthy
        SDL_SetRenderDrawColor(r, ((p&1) ? max(255/(((p&252)>>7)+1), 0) : 0)*(p&(1<<1))>>1, ((p&1) ? 0 : 255/(((p&252)>>7)+1))*(p&(1<<1))>>1, 255-((p&252)>>2)*(p&(1<<1))>>1, 255);
        SDL_Rect rect;
        rect.x = (p>>8)&2047;
        rect.y = p>>19;
        rect.w = 3;
        rect.h = 3;
        SDL_RenderFillRect(r, &rect);
        
        SDL_RenderDrawPoint(r, (p>>8)&2047, p>>19);
        
    }

    //SDL_UpdateTexture(txt, NULL, people_pixels, W);
    //SDL_RenderCopy(r, txt, NULL, NULL);

    global_framecount++;
}

void eventloop() {
    while(quit != 1) {
        SDL_Event e;
        while(SDL_PollEvent(&e) != NULL) {
            switch(e.type) {
                case SDL_QUIT:
                    quit = 1;
                    break;
            }
        }
        SDL_SetRenderDrawColor(r, 0, 0, 0, 0);
        SDL_RenderClear(r);
        render();
        SDL_RenderPresent(r);
    }

    SDL_DestroyRenderer(r);
    SDL_DestroyWindow(w);
    w = NULL;
    r = NULL;
    SDL_Quit();
}

int main(int argc, char** argv) {
    srand((unsigned)time(NULL));

    //init sdl
    if(init_sdl() < 0) {
        perror("cannot initialize SDL!");    
        return -1;
    }

    people = (unsigned int*)malloc(sizeof(unsigned int)*N);
    people_pixels = (unsigned int*)malloc(sizeof(unsigned int)*N);
    for(int i = 0; i < N; i++) {
        // everyone starts from random ages between 1 and 50
        // x and y coordinates are from 0-1920 respectively, they fit in 11 bits
        people[i] = (int)((float)(rand()/(float)RAND_MAX)*W.0f);
        people[i] <<= 11;
        people[i] += (int)((float)(rand()/(float)RAND_MAX)*H.0f);
        people[i] <<= 6;
        people[i] += (int)((float)(rand()/(float)RAND_MAX)*40.0f);
        people[i] <<= 2;
        people[i] += 2;
        //people[i] += 1;
    }
    for(int i = 0; i < 1; i++) {
        people[i] |= 1;
    }
    cudaMalloc((void**)&d_people, sizeof(unsigned int)*N);
    cudaMalloc((void**)&d_people_pixels, sizeof(unsigned int)*N);
    
    cudaMemcpy(d_people, people, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);

    eventloop();

    free(people);
    free(people_pixels);
    cudaFree(d_people);
    cudaFree(d_people_pixels);
    people = NULL;
    d_people = NULL;
    return 0;
}

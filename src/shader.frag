#version 330 core
in vec4 colour;
in vec4 gl_FragCoord;   // but actually use UVs hey
in vec2 uv;
flat in uint fs_mode;

out vec4 frag_colour;

#define PI 3.1415926535897932384626433832795
#define ROOT2INV 0.70710678118654752440084436210484903928

uniform sampler2D tex;
uniform float time;

float rand(float n){return fract(sin(n) * 43758.5453123);}

float noise(float p){
	float fl = floor(p);
  float fc = fract(p);
	return mix(rand(fl), rand(fl + 1.0), fc);
}

float f1d(float p){
  return 
    1.000 * noise(p) +
    0.500 * noise(p*2.0 + 15123.34521) +
    0.250 * noise(p*4.0 + 13418.23523) +
    0.125 * noise(p*8.0 + 19023.52627) /
    1.875
    ;
}

// between 0 and 1
// then its -ln for mountain mode

float slowstart(float t) {
    return 1.0 - (1.0 - t)*(1.0 - t);
}
float slowstop(float t) {
    return t*t;
}

float quadraticInOut(float t) {
  float p = 2.0 * t * t;
  return t < 0.5 ? p : -p + (4.0 * t) - 1.0;
}
float exponentialInOut(float t) {
  return t == 0.0 || t == 1.0
    ? t
    : t < 0.5
      ? +0.5 * pow(2.0, (20.0 * t) - 10.0)
      : -0.5 * pow(2.0, 10.0 - (t * 20.0)) + 1.0;
}

void main() {
    switch(fs_mode) {
        case 0u:
        frag_colour = colour;
        break;
        case 1u:
        frag_colour = texture(tex, uv) * colour;
        break;
        default:
        frag_colour = vec4(0., 1., 0., 1.);
        break;
        case 2u:
        // 0.325
        float x = (0.5-uv.x);
        float y = (0.5-uv.y);
        float r = sqrt(x*x + y*y);
        float d_circle = abs(r - 0.325);
        float o = d_circle / (0.5 - 0.325);


        //float r = sqrt((0.5-uv.x)*(0.5-uv.x)+(0.5-uv.y)*(0.5-uv.y));
        // float mask = r < 0.15 ? 0.0 : r > 0.5 ? 0.0 : 1.0;
        
        float oo = (o*o);
        float oooo = (oo*oo);
        float mask = o > 1.0 ? 0.0 : 1.0 - oooo*oooo*oooo*oooo;
        float theta = atan(y, x);
        float theta1 = (theta + 0.66*PI*time);
        float theta2 = (theta + -0.5*PI*time);
        float t1 = mod(theta1, 2*PI);
        t1 -= PI;
        t1 = abs(t1) / PI;

        
        float t2 = mod(theta2, 2*PI);
        t2 -= PI;
        t2 = abs(t2) / PI;



        float inness = 1.0-o;
        // t1 *= inness;
        // t2 *= inness;
        
        float t = max(t1, t2);
        //float t = abs((max(mod(theta1, 2*PI), mod(theta2, 2*PI)) - PI) / (PI));
        t=t*t;
        t *= inness;
        frag_colour = mask*mix(colour, vec4(1., 1., 1., 1.), t);
        //frag_colour = mix(colour, vec4(0., 0., 0., 1.), exponentialInOut(t * 4.0));
        break;
        case 1000u:
        float h1 = 0.6 -0.3 * log(f1d(uv.x * 3 + time * 0.1));
      
        float h2 = 0.5 -0.2 * log(f1d(1238+(uv.x * 4) + (12238+time) * 0.2));
        float h3 = 0.4 -0.1 * log(f1d(7633+(uv.x * 5) + (55645+time) * 0.3));

        h1 = 1 - h1;
        h2 = 1 - h2;
        h3 = 1 - h3;

        // float h = 0.4 + 0.2 * f1d(uv.x * 10 + time * 1);
        if (uv.y > h3) {
          frag_colour = vec4(0.55, 0.39, 0.25, 1.0);
        } else if (uv.y > h2) {
          frag_colour = vec4(0.53, 0.33, 0.25, 1.0);
        } else if (uv.y > h1){
          frag_colour = vec4(0.5, 0.3, 0.25, 1.0);
        } else {
          frag_colour = vec4(0.55, 0.55, 0.9, 1.0);
        }
        break;
        case 1001u:
        // or if the geometry, it actually splits into 4 and they swap places, or flip and flip UVs
        // for transition diamonds shrink revealing next thing
        // if L1 dist > t - tlast kind of thing
        // the flag in crash team racing: with some kind of domain warping applied
        theta = PI/4;
        float up = cos(theta) * uv.x - sin(theta) * uv.y;
        float vp = sin(theta) * uv.x + cos(theta) * uv.y;

        up = up + time * 0.02;
        vp = vp + time * 0.0015;

        up *= 5.0;
        vp *= 5.0;

        up = mod(up, 1.0);
        vp = mod(vp, 1.0);

        if (up < 0.5 ^^ vp < 0.5) {
          frag_colour = vec4(0.6, 0.1, 0.6, 1.0);
        } else {
          frag_colour = vec4(0.3, 0.1, 0.6, 1.0);
        }

        break;
    }
}
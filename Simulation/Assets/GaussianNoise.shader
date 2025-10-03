// Based on:
// https://github.com/thomas-moulard/gazebo-deb/blob/master/media/materials/programs/camera_noise_gaussian_fs.glsl
// and
// https://www.alanzucconi.com/2015/07/08/screen-shaders-and-postprocessing-effects-in-unity3d/

Shader "Custom/GaussianNoise" {
    Properties {
        _MainTex ("Base (RGB)", 2D) = "white" {}
        _offsets ("Random Vector for Noise", Vector) = (0, 0, 0, 1)
        _mean ("Mean of the Applied Gaussiam Noise", Float) = 0
        _stddev ("Standard Deviation of the Applied Gaussiam Noise", Float) = 0.007
    }
    SubShader {
        Pass {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex;
            uniform float4 _offsets;
            uniform float _mean;
            uniform float _stddev;

            #define PI 3.14159265358979323846264

            struct v2f 
			{
				float4 pos : SV_POSITION;
				float2 uv:TEXCOORD0;
			};

			//Vertex Shader
			v2f vert(appdata_base v) 
			{
				v2f o;
				o.pos = UnityObjectToClipPos(v.vertex);
				o.uv = ComputeScreenPos(o.pos);
				o.uv.y = 1 - o.uv.y;

				return o;
			}

            float rand(float2 co)
            {
                // This one-liner can be found in many places, including:
                // http://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
                // I can't find any explanation for it, but experimentally it does seem to
                // produce approximately uniformly distributed values in the interval [0,1].
                float r = frac(sin(dot(co, float2(12.9898,78.233))) * 43758.5453);

                // Make sure that we don't return 0.0
                if(r == 0.0)
                    return 0.000000000001;
                else
                    return r;
            }

            float4 gaussrand(float2 co)
            {
                // Box-Muller method for sampling from the normal distribution
                // http://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
                // This method requires 2 uniform random inputs and produces 2 
                // Gaussian random outputs.  We'll take a 3rd random variable and use it to
                // switch between the two outputs.

                float U, V, R, Z;
                // Add in the CPU-supplied random offsets to generate the 3 random values that
                // we'll use.
                U = rand(co + float2(_offsets.x, _offsets.x));
                V = rand(co + float2(_offsets.y, _offsets.y));
                R = rand(co + float2(_offsets.z, _offsets.z));
                // Switch between the two random outputs.
                if(R < 0.5)
                Z = sqrt(-2.0 * log(U)) * sin(2.0 * PI * V);
                else
                Z = sqrt(-2.0 * log(U)) * cos(2.0 * PI * V);

                // Apply the stddev and mean.
                Z = Z * _stddev + _mean;

                // Return it as a vec4, to be added to the input ("true") color.
                return float4(Z, Z, Z, 0.0);
            }

            float4 frag(v2f i) : COLOR {
                float4 c = tex2D(_MainTex, i.uv);
                
                return clamp(c + gaussrand(i.pos.xy), 0.0, 1.0);
            }
            ENDCG
        }
    }
}
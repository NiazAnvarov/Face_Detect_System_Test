using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Face_Detect_System_Test
{
    internal class FacesDetect
    {

        private FaceDetectorYN _detector;

        public Mat DetectFaces(Mat frame)
        {

            _detector = new FaceDetectorYN(
                    model: "H:\\face_detection_yunet_2023mar.onnx",
                    config: string.Empty,
                    inputSize: new System.Drawing.Size(640, 480),
                    scoreThreshold: 0.9f,
                    nmsThreshold: 0.3f,
                    topK: 5000,
                    backendId: Emgu.CV.Dnn.Backend.Default,
                    targetId: Target.Cpu);

            var faces = new Mat();
            _detector.Detect(frame, faces);

            return faces;
        }

        public void Dispose()
        {
            _detector?.Dispose();
        }

    }
}

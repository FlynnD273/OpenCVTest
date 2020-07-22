package com.example.opencvtest;

import android.graphics.Bitmap;
import android.graphics.Color;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ImageMapper {
    private ColorBlobDetector detector;

    public ColorBlobDetector getDetector() {
        return detector;
    }

    public void setDetector(ColorBlobDetector mDetector) {
        this.detector = mDetector;
    }

    private Mat source;

    public Mat getSource() {
        return source;
    }

    public void setSource(Mat source) {
        this.source = source;
    }

    private boolean debug;

    public boolean isDebug() {
        return debug;
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }


    private Bitmap img;

    public Bitmap getImg() {
        return img;
    }

    public void setImg(Bitmap img) {
        this.img = img;
    }

    public ImageMapper(boolean d) {
        detector = new ColorBlobDetector();
        source = new Mat();
        debug = d;
    }

    public ImageMapper(Bitmap image, boolean d) {
        this(d);
        img = image;
    }

    public void drawFromFourContours() {
        List<MatOfPoint> contours = findContours(source);
        if (contours.size() > 3) {
            drawImage(contours);
        }
    }

    public void drawFromContour() {
        List<MatOfPoint> contours = findContours(source);

        //Find contours that have around four vertices
        for (MatOfPoint m : contours) {
            MatOfPoint2f mat = new MatOfPoint2f(m.toArray());
            double peri = Imgproc.arcLength(mat, true);
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(mat, approx, 0.05 * peri, true);
            Point[] points = approx.toArray();
            if (points.length > 3 && points.length < 8) {
                drawImage(points);
            }
        }
    }

    /**
     * Draw an image between four points
     *
     * @param p The four Points to draw between
     */
    private void drawImage(Point[] p) {
        drawImage(p[0], p[1], p[2], p[3]);
    }

    /**
     * Draw an image between four points
     *
     * @param contours The four contours to draw between
     */
    private void drawImage(List<MatOfPoint> contours) {
        //Get the center point of the first four contours
        Rect r = Imgproc.boundingRect(contours.get(0));
        Point a = new Point(r.x + r.width / 2, r.y + r.height / 2);
        r = Imgproc.boundingRect(contours.get(1));
        Point b = new Point(r.x + r.width / 2, r.y + r.height / 2);
        r = Imgproc.boundingRect(contours.get(3));
        Point c = new Point(r.x + r.width / 2, r.y + r.height / 2);
        r = Imgproc.boundingRect(contours.get(2));
        Point d = new Point(r.x + r.width / 2, r.y + r.height / 2);

        drawImage(a, b, c, d);
    }

    /**
     * Draw an image between four points
     *
     * @param a Upper Left
     * @param b Upper Right
     * @param c Lower Right
     * @param d Lower Left
     */
    private void drawImage(Point a, Point b, Point c, Point d) {
        List<MatOfPoint> draw = new ArrayList<>();
        draw.add(new MatOfPoint(a, b, c, d));
        //Draw bounding box
        if (debug) {
            Imgproc.polylines(source, draw, true, detector.getmBlobRgbColor(), 5);
        }

        //Copy points into an array
        Point[] p = draw.get(0).toArray();

        //Average of all four points to get center
        Point center = new Point(0, 0);
        for (Point point : p) {
            center.x += point.x;
            center.y += point.y;
        }
        center.x /= p.length;
        center.y /= p.length;

        //Sort clockwise around center point
        Arrays.sort(p, (first, second) -> less(first, second, center) ? 1 : -1);

        b = p[0];
        c = p[1];
        d = p[2];
        a = p[3];

        //Draw corners
        if (debug) {
            Scalar circleCol = new Scalar(255, 0, 255, 255);
            Imgproc.circle(source, a, 5, circleCol);
            Imgproc.circle(source, b, 5, circleCol);
            Imgproc.circle(source, c, 5, circleCol);
            Imgproc.circle(source, d, 5, circleCol);
        }

        //Draw the image
        mapImage(a, b, c, d, img, source);
    }

    /**
     * Distorts an image to fit four points. Points must be sorted
     *
     * @param a   Upper Left
     * @param b   Upper Right
     * @param c   Lower Right
     * @param d   Lower Left
     * @param img Bitmap to distort
     */
    private void mapImage(Point a, Point b, Point c, Point d, Bitmap img, Mat drawTo) {
        int pixel;
        int h = img.getHeight();
        int w = img.getWidth();
        for (int y = 0; y < h; y++) {
            //Get the corners for the row
            Point ulc = lerp(a, d, (float) y / h);
            Point urc = lerp(b, c, (float) y / h);
            Point llc = lerp(a, d, (float) (y + 1) / h);
            Point lrc = lerp(b, c, (float) (y + 1) / h);

            for (int x = 0; x < w; x++) {
                //Get the pixel color
                pixel = img.getPixel(x, y);

                //Only draw if pixel is not transparent
                if (Color.alpha(pixel) > 10) {
                    //Convert to Scalar
                    Scalar col = new Scalar(Color.red(pixel), Color.green(pixel), Color.blue(pixel));

                    //Get the corners for the pixel
                    Point ul = lerp(ulc, urc, (double) x / w);
                    Point ll = lerp(llc, lrc, (double) x / w);
                    Point ur = lerp(ulc, urc, (double) (x + 1) / w);
                    Point lr = lerp(llc, lrc, (double) (x + 1) / w);

                    //Fill the pixel
                    List<MatOfPoint> matPointList = new ArrayList<>();
                    matPointList.add(new MatOfPoint(ul, ur, lr, ll));
                    Imgproc.fillPoly(drawTo, matPointList, col);
                }
            }
        }
    }

    /**
     * Process the Mat mRgba
     *
     * @return List<MatOfPoint> that contains all contours found
     */
    private List<MatOfPoint> findContours(Mat src) {
        detector.process(src);
        List<MatOfPoint> contours = detector.getContours();
        return contours;
    }

    /**
     * Compares two Points around a center Point
     *
     * @param a      First Point
     * @param b      Second Point
     * @param center Point to compare to
     * @return True is a is to the left of b, restarting at pi/2
     */
    private boolean less(Point a, Point b, Point center) {
        //Eliminate easy cases
        if (a.x - center.x >= 0 && b.x - center.x < 0)
            return true;
        if (a.x - center.x < 0 && b.x - center.x >= 0)
            return false;
        if (a.x - center.x == 0 && b.x - center.x == 0) {
            if (a.y - center.y >= 0 || b.y - center.y >= 0)
                return a.y > b.y;
            return b.y > a.y;
        }

        // compute the cross product of vectors (center -> a) x (center -> b)
        double det = (a.x - center.x) * (b.y - center.y) - (b.x - center.x) * (a.y - center.y);
        if (det < 0)
            return true;
        if (det > 0)
            return false;

        // points a and b are on the same line from the center
        // check which point is closer to the center
        double d1 = (a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (a.y - center.y);
        double d2 = (b.x - center.x) * (b.x - center.x) + (b.y - center.y) * (b.y - center.y);
        return d1 > d2;
    }

    /**
     * Linear interpolation for Points
     *
     * @param a   First Point
     * @param b   Second Point
     * @param amt Double between 0 and 1. 0 returns a, 1 returns b
     * @return
     */
    private Point lerp(Point a, Point b, double amt) {
        double x = lerp(a.x, b.x, amt);
        double y = lerp(a.y, b.y, amt);
        return new Point(x, y);
    }

    /**
     * Linear interpolation
     *
     * @param a   First value
     * @param b   Second value
     * @param amt Double between 0 and 1. 0 returns a, 1 returns b
     * @return
     */
    private double lerp(double a, double b, double amt) {
        return a * amt + b * (1 - amt);
    }
}

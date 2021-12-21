package com.dong_pham.trafficsignrecognition.views;

import static androidx.core.content.ContextCompat.checkSelfPermission;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.app.SearchManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import com.dong_pham.trafficsignrecognition.R;
import com.dong_pham.trafficsignrecognition.ml.ModelUnquant;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;




public class HomeActivity extends AppCompatActivity {

    TextView result;
    ImageView imageView;
    Button picture, select,btnSearch;
    int imageSize = 224 ;
    Intent intent;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        select = findViewById(R.id.select);
        btnSearch = findViewById(R.id.search);

        picture.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                //Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }

            }


        });

        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*") ;
                startActivityForResult(intent, 200);

            }
        });
        btnSearch.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_WEB_SEARCH);
                String term = result.getText().toString();
                intent.putExtra(SearchManager.QUERY, term);
                startActivity(intent);
            }
        });
    }


    public void classifyImage(Bitmap image){
        try {
            @NonNull ModelUnquant model = ModelUnquant.newInstance(peekAvailableContext().getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // get 1D array of 224 * 224 pixels in image
            int [] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            // iterate over pixels and extract R, G, and B values. Add to bytebuffer.
            int pixel = 0;
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for(int i = 0; i < confidences.length; i++){
                if(confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {" Tốc độ tối đa cho phép (60km/h)",
                    " Tốc độ tối đa cho phép (80km/h)",
                    " Tốc độ tối đa cho phép (100km/h)",
                    " Tốc độ tối đa cho phép (120km/h)",
                    " Cấm hai xe cơ giới vượt nhau",
                    " Giao nhau với đường ưu tiên",
                    " Dừng lại",
                    " Đường cấm",
                    " Cấm phương tiện > 3.5 tấn",
                    " Cấm vào",
                    " Nguy hiểm",
                    " Nhiều chỗ ngoặt nguy hiểm liên tiếp",
                    " Đường có ổ gà, lồi lõm",
                    " Đường trơn",
                    " Đường bị hẹp về phía phải",
                    " Công trường",
                    " Tín hiệu giao thông",
                    " Đường dành cho người đi bộ",
                    " Hiệu lệnh rẽ phải",
                    " Hiệu lệnh rẽ trái",
                    "  Hướng đi thẳng phải theo",
                    " Chỉ được đi thẳng và rẽ phải",
                    " Chỉ được đi thẳng và rẽ trái",
                    " Đi vòng sang trái",
                    " Giao nhau chạy theo vòng xuyến"
            };
            result.setText(classes[maxPos]);

            String s = "";
            for(int i = 0; i < classes.length; i++){
                s += String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100);
            }
            // confidence.setText(s);


            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }


//    @Override
//    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
//        if(count==1){
//            if (requestCode == 1 && resultCode == RESULT_OK) {
//                Bitmap image = (Bitmap) data.getExtras().get("data");
//                int dimension = Math.min(image.getWidth(), image.getHeight());
//                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
//                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
//                imageView.setImageBitmap(image);
//                imageView.setImageURI(data.getData());
//
//            }
//            super.onActivityResult(requestCode, resultCode, data);
//
//        }
//        else if (count ==0){
//            Uri uri = data.getData();
//
//            try {
//                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.peekAvailableContext().getContentResolver(), uri);
//                int dimension = Math.min(bitmap.getWidth(), bitmap.getHeight());
//                bitmap = ThumbnailUtils.extractThumbnail(bitmap, dimension, dimension);
//                bitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, false);
//                imageView.setImageBitmap(bitmap);
//                classifyImage(bitmap);
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//            super.onActivityResult(requestCode, resultCode, data);
//        }
    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            classifyImage(image);
        } else if (requestCode == 200) {
            // super.onActivityResult(requestCode, resultCode, data);
            Uri uri = data.getData();
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.peekAvailableContext().getContentResolver(), uri);
                int dimension = Math.min(bitmap.getWidth(), bitmap.getHeight());
                bitmap = ThumbnailUtils.extractThumbnail(bitmap, dimension, dimension);
                bitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, false);
                imageView.setImageBitmap(bitmap);
                classifyImage(bitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }


}


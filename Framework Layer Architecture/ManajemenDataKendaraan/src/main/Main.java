package main;

import java.util.ArrayList;
import java.util.Scanner;

import adapter.KendaraanAdapter;
import model.Kendaraan;
import singleton.KendaraanManager;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		KendaraanManager manager = KendaraanManager.getInstance();
		Scanner sc = new Scanner(System.in);
		
		String merek, tipe;
		int tahun;	
		int pilih;
		
		do {
			System.out.println("1. Tambah Kendaraan");
			System.out.println("2. Exit");
			System.out.print("Pilih Menu : ");
			pilih = sc.nextInt(); sc.nextLine();
			
			if(pilih == 1) {
//				Length Merek 3 - 20 (Inclusive)
				do {
					System.out.print("Masukkan Merek : ");
					merek = sc.nextLine();
				} while (merek.length() < 3 || merek.length() > 20);
						
				System.out.print("Masukkan Tahun : ");
				tahun = sc.nextInt(); sc.nextLine();
				
//				Tipe harus "Mobil" atau "Motor" Equals
				do {
					System.out.print("Masukkan Tipe : ");
					tipe = sc.nextLine();
					if(!(tipe.equals("Mobil") || tipe.equals("Motor"))) {
						System.out.println("Tipe harus Mobil atau Motor");
					}
				} while (!(tipe.equals("Mobil") || tipe.equals("Motor")));
				
				manager.tambahKendaraan(merek, tahun, tipe);
			} else if (pilih == 2) {
				System.out.println("Selamat Tinggal");
			} else {
				System.out.println("Pilihan menu tidak tersedia");
			}
			
		} while (pilih != 2);
		
//		manager.tambahKendaraan("Toyoto", 2025, "mobil");
//		manager.tambahKendaraan("Hondo", 2023, "motor");
		
		ArrayList<Kendaraan> kendaraans = manager.ambilKendaraan();
		// foreach
		for (Kendaraan kendaraan : kendaraans) {
			System.out.println(KendaraanAdapter.formatKendaraan(kendaraan));
		}
	}

}

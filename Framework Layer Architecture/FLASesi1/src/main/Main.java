package main;

import java.util.Scanner;

import model.Dosen;

public class Main {

	
	public static void main(String[] args) {
		
		String NamaP, Desc;
		int JamMasuk, JamKeluar, gaji;
		
		Scanner sc = new Scanner(System.in);	//Make Inputer
		
		System.out.println("Masukkan nama kerjaan: ");
		NamaP = sc.nextLine();
			
		System.out.println("Masukkan Jam Masuk: ");
		JamMasuk = sc.nextInt(); sc.nextLine();	//Integer
		
		System.out.println("Masukkan Jam Keluar: ");
		JamKeluar = sc.nextInt(); sc.nextLine(); //Integer
		
		System.out.println("Masukkan Deskripsi: ");
		Desc = sc.nextLine();
		
		System.out.println("Masukkan Gaji: ");
		gaji = sc.nextInt(); sc.nextLine(); //Integer
		
	
		
		
		Dosen Dosen1 = new Dosen(NamaP, JamMasuk, JamKeluar, Desc, gaji);
		
		
		Dosen1.BerangkatKerja();
		
	}
}

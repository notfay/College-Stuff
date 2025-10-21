package main;

import model.*;
import proxy.*;
import template.*;
import java.util.*;

public class Main {
    private HashMap<String, Kendaraan> db = new HashMap<>();
    private InfoKendaraan infoKendaraan = new InfoProxy();
    private Scanner sc = new Scanner(System.in);
    
    public Main() {
        int choice = 0;
        while (true) {
            // Simple menu display
            System.out.println("--- SOroom Vehicle Purchase System ---");
            System.out.println("1. Beli Kendaraan");
            System.out.println("2. Lihat Pembelian Kendaraan");
            System.out.println("3. Exit");
            System.out.print("Choose menu [1-3]: ");
                        
            if (choice == 1) {
                beliKendaraan();
            } else if (choice == 2) {
                lihatPembelian();
            } else if (choice == 3) {
                System.out.println("Thank you for using SOroom!");
                break;
            } else {
                System.out.println("Invalid choice.");
            }
        }
    }

    private void beliKendaraan() {
        // Get name with validation
        String nama;
        do {
            System.out.print("Masukkan Nama: ");
            nama = sc.nextLine();
        } while (nama.isEmpty());

        // Get price with validation
        int harga = 0;
        while (harga <= 0) {
            System.out.print("Masukkan Harga: ");
            try {
                harga = Integer.parseInt(sc.nextLine());
                if (harga <= 0) System.out.println("Harga harus lebih dari 0.");
            } catch (Exception e) {
                System.out.println("Masukkan angka yang valid.");
            }
        }

        // Get type with validation
        String tipe;
        while (true) {
            System.out.print("Tipe Kendaraan [Mobil/Motor]: ");
            tipe = sc.nextLine();
            if (tipe.equals("Mobil") || tipe.equals("Motor")) break;
            System.out.println("Tipe tersebut tidak dijual!");
        }

        // Generate code and create objects
        String code = generateCode();
        Kendaraan kendaraan;
        BeliKendaraan beli;
        
        if (tipe.equals("Mobil")) {
            kendaraan = new Mobil(nama, harga);
            beli = new BeliMobil(nama);
        } else {
            kendaraan = new Motor(nama, harga);
            beli = new BeliMotor(nama);
        }
        
        db.put(code, kendaraan);
        beli.prosesPembelian();
        System.out.println("Pembelian berhasil. Code kendaraanmu: " + code);
    }

    private void lihatPembelian() {
        System.out.print("Masukkan nama kendaraan: ");
        String nama = sc.nextLine();

        boolean ditemukan = false;
        for (Kendaraan k : db.values()) {
            if (k.getNama().equals(nama)) {
                System.out.println("Info: " + infoKendaraan.getInfo(k));
                ditemukan = true;
            }
        }

        if (!ditemukan) {
            System.out.println("Kendaraan tidak ditemukan dalam pembelian.");
        }
    }

    private String generateCode() {
        String chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        StringBuilder code = new StringBuilder();
        Random rand = new Random();
        for (int i = 0; i < 5; i++) {
            code.append(chars.charAt(rand.nextInt(chars.length())));
        }
        return code.toString();
    }
    
    public static void main(String[] args) {
        new Main();
    }
}
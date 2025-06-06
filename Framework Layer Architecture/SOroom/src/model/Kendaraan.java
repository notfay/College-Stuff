package model;

public abstract class Kendaraan {
    private String nama;
    private int harga;
    public abstract String getTipe();
    
    public Kendaraan(String nama, int harga) {
        this.nama = nama;
        this.harga = harga;
    }

    

    public String getNama() {
        return nama;
    }

    public int getHarga() {
        return harga;
    }
}

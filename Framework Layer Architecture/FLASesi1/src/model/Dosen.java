package model;

public class Dosen extends Pekerjaan{ //CHild Class
	
	private int Gaji;
	
	//Constructor with Gaji + Default
	
	public Dosen(String namaPekerjaan, int jamMasuk, int jamKeluar, String deskripsi, int gaji) {
		super(namaPekerjaan, jamMasuk, jamKeluar, deskripsi);
		Gaji = gaji;
	}

	public int getGaji() {
		return Gaji;
	}
	
	//Getter setter for Gaji
	
	public void setGaji(int gaji) {
		Gaji = gaji;
	}

	@Override
	public void BerangkatKerja() {
		System.out.println("Berangkat " + getJamMasuk());
	}
	
	

	
}

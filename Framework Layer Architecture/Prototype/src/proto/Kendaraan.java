package proto;

public class Kendaraan implements Cloneable {
	private String merek;
	private String Warna;
	
	//Constructor
	public Kendaraan(String merek, String warna) {
		super();
		this.merek = merek;
		Warna = warna;
	}
	
	//Overloading -> Nama function/method sama tapi beda parameter
//	public Kendaraan(Kendaraan kendaraanClone) {
//		this.merek = kendaraanClone.merek;
//		this.Warna = kendaraanClone.Warna;
//	}
//	
	
	
	
	
	//Get Merek
	public void getInfo() {
		System.out.println("Merek " + merek + " Warna " + Warna);
	}
	
	public void gantiMerk(String merek) {
		this.merek = merek;
	}
	
//	public Kendaraan clone( ) {
//		return new Kendaraan(this);
//	}
	
	
	@Override 
	public Kendaraan clone() {
		try {
			return (Kendaraan) super.clone();
		} catch (CloneNotSupportedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	
	
}
